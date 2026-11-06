# Experiment 1 — Unpinned co-run of games and AI workloads

Experiment 1 asks a single question:

> When a video game is already using the NVIDIA GPU, how do frame rate, GPU utilisation, power, temperature, and CPU use change if an AI inference job is running at the same time — either **on that same GPU**, or **entirely on the CPU**?

There is **no CPU-core pinning** in this experiment. The operating system scheduler places threads wherever it wants. Later experiments (not covered here) add pinning and game-only CPU demand measurements.

---

## 1. What is being tested

### 1.1 Four games

Every combination runs the same four titles. Each is launched in a **repeatable, non-interactive** mode so a human does not have to play.

| Game | How it is driven | Notes |
|------|------------------|--------|
| **Xonotic** | Plays a fixed demo (`demos/rnd_demo.dem`) at 1920×1080, uncapped FPS (`+cl_maxfps 0`) | Local install path is hardcoded in `run_game.py` |
| **glmark2** | Window at 1920×1080 | System `glmark2` binary |
| **Luanti** | `--go --random-input` (starts a world and injects random input) | Local install path is hardcoded in `run_game.py` |
| **SuperTuxKart** | Demo mode: 8 karts, 100 laps, 1920×1080 | System `supertuxkart` binary |

The game is **always** forced onto the discrete NVIDIA GPU via PRIME render offload:

- `__NV_PRIME_RENDER_OFFLOAD=1`
- `__GLX_VENDOR_LIBRARY_NAME=nvidia`

So in both placement modes below, the **game stays on the GPU**. Only the AI job moves.

### 1.2 Three AI workloads

| Name | Model and data | How a “request” is defined |
|------|----------------|----------------------------|
| **cnn** | CIFAR-10 ResNet20 (`chenyaofo/pytorch-cifar-models`, pretrained). The CIFAR-10 test set (10,000 images) is preloaded as **10 batches** (nine of 1024 images and one of 784). | One forward pass of a randomly chosen batch, with `torch.cuda.synchronize()` when on GPU so latency is real wall time |
| **tinybert** | `prajjwal1/bert-tiny` on GLUE SST-2 (first 1000 validation sentences) | One forward pass of up to 32 randomly sampled sentences, tokenised to max length 128 |
| **qwen** | `qwen:0.5b` served by a local **Ollama** process (`http://localhost:11434`). Prompts are random instructions from Databricks Dolly-15k | One HTTP generate call (non-streaming). Timeout 120 s. Token count comes from Ollama’s `eval_count` |

CNN and TinyBERT are ordinary PyTorch processes started by `run_ai_workload.py`. Qwen is different: that script only sends HTTP requests; the actual inference runs inside Ollama’s own server.

### 1.3 Two placement modes

| Mode | Folder / combo name suffix | What the AI does | What the game does |
|------|----------------------------|------------------|--------------------|
| **Shared GPU** | `_gpu` | AI runs on the NVIDIA GPU | NVIDIA GPU via PRIME |
| **AI on CPU** | `_cpu` | AI runs on the CPU | NVIDIA GPU via PRIME (unchanged) |

The first mode is contention on the GPU (and whatever CPU the game and Python still use). The second mode is contention mainly on the CPU, while the GPU is left to the game.

**CNN and TinyBERT:** `run_batch.py` creates both modes automatically by passing `--device gpu` or `--device cpu` to `run_ai_workload.py`.

**Qwen (Ollama):** the two modes are **not** created by `run_batch.py`. It only creates folders named `qwen_<intensity>_ollama`, and the `--device` flag does not move Ollama. Placement is whatever the **already running** Ollama server was started with. The procedure is therefore done in two passes:

1. **Pass 1 — Ollama on the GPU (shared with the game).** Start Ollama normally and leave it running for the whole Qwen set:

   ```
   ollama serve
   ```

   Run the Qwen sweep and let it finish (see §3.3).

2. **Pass 2 — Ollama off the GPU (AI on CPU).** Stop the running Ollama server, restart it so that inference runs on the CPU, and run the Qwen sweep again:

   ```
   CUDA_VISIBLE_DEVICES=-1 OLLAMA_LLM_LIBRARY=cpu ollama serve
   ```

Leave each server running for the whole pass. Switching mid-sweep would mix GPU and CPU traces in the same `_ollama` folders.

Because both passes use the same combination names (`qwen_<intensity>_ollama`), the second pass would overwrite or skip the first. Keep the passes apart by giving each its own output folders, for example:

```bash
# Pass 1: Ollama on GPU
python run_batch.py --models qwen --results-dir results_ollama_gpu --per-core-dir per_core_ollama_gpu

# Pass 2: Ollama on CPU (after restarting Ollama in CPU placement)
python run_batch.py --models qwen --results-dir results_ollama_cpu --per-core-dir per_core_ollama_cpu
```

Each results folder has its own `state.json`, so the second pass starts fresh.

### 1.4 Request intensity (how hard the AI is driven)

Between successive AI requests the script sleeps:

| Intensity | Sleep after each request |
|-----------|--------------------------|
| `high` | 0.05 s |
| `mid` | 0.5 s |
| `low` | 2.0 s |
| `idle` | no requests; the process just sleeps 1 s at a time (model may still sit in memory / on the GPU) |

The default sweep uses **all four** intensities.

### 1.5 Repeats

For **each** combination of (AI model × intensity × placement mode):

1. The AI job is started once and left running.
2. Each of the **four games** is run **5 times**.
3. Each game run lasts **240 seconds** (4 minutes) by default.

Default sweep size (one run of `python run_batch.py`):

- cnn: 4 intensities × 2 modes = 8 combinations
- tinybert: 8 combinations
- qwen: 4 intensities × 1 (`ollama`) = 4 combinations
- **20 combinations** × 4 games × 5 runs = **400 game runs**

Running the Qwen sweep a second time for Ollama on CPU (§1.3) adds 4 more combinations and 80 more game runs.

**Expected duration.** Each combination takes about 300 s warmup + 240 s baseline + 20 × (240 s + ~10 s for MangoHud flush and shutdown) ≈ 5,540 s, or roughly **1 h 32 m**. The full 20-combination sweep is therefore roughly **31 hours**, and cooldown waits come on top of that (the time estimate printed by `run_batch.py` does not include them).

---

## 2. Temperature control

Temperature is used so that one hot run does not bias the next. There are **two different targets**.

### 2.1 Cooldown *between game runs* (inside one combination)

After the AI is loaded, the controller samples GPU temperature with `nvidia-smi` for **240 s** (default `--baseline-duration`) while the AI is already running and **no game** is running. The **mean** of those samples is the **baseline**.

The target passed into `run_game_batch.py` is:

```text
cooldown_target = ceil(baseline_avg + temp_margin)
```

- `--temp-margin` default is **2 °C**
- The value is rounded **up** to a whole degree because `run_game_batch.py` only accepts an integer `--target-temp`

Between every game run except the last in that combination, the batch script waits until:

```text
GPU temperature ≤ cooldown_target
```

Polling is every **5 s**. If the GPU has not reached the target within **600 s** (`--cooldown-timeout`), the next run starts anyway.

This target is **not** a fixed 45 °C. It tracks how hot the AI alone makes the GPU, plus 2 °C, so cooldown returns the machine to “AI-only steady state”, not to a cold idle.

The measured `baseline_avg` and `cooldown_target` are written into that combination’s `meta.json`.

### 2.2 Cooldown *between combinations*

After a combination finishes (AI stopped, logs collected), the controller waits until the GPU is ≤ **45 °C** (`--idle-temp`, default 45) before starting the next model/intensity/mode. Same 5 s poll and 600 s timeout.

When `run_game_batch.py` is used **by itself** (not through `run_batch.py`), its own default `--target-temp` is **45 °C**.

---

## 3. Inputs you must provide

### 3.1 Machine and software

- Linux (per-core logging reads `/proc/stat`)
- NVIDIA driver so `nvidia-smi` can read temperature, utilisation, power, clocks, and VRAM
- **MangoHud** on `PATH` (wraps the game and writes FPS / frame-time CSVs)
- `timeout` on `PATH`. The preflight check in `run_batch.py` requires the binary to exist, although `run_game.py` itself does not wrap the game with it
- Python with `torch`, `torchvision`, `transformers`, `datasets`, `requests`
- CUDA visible to that Python if any `_gpu` combinations are requested (otherwise those combinations would silently fall back to CPU; preflight refuses to start in that case)
- For Qwen: Ollama listening on `http://localhost:11434` with model `qwen:0.5b` pulled, started in the intended placement **before** each Qwen pass (see §1.3)

### 3.2 Game binaries

Paths are in `GAMES_CONFIG` inside `run_game.py`. **Xonotic and Luanti currently point at a local disk** (`/run/media/tanish/...`). On another machine these paths would have to be changed in `run_game.py`. glmark2 and SuperTuxKart are expected as system commands.

Launch arguments (resolution, demo file, random input, demo laps) are part of the experimental **input**: they define the visual load. Do not change them if you want comparable traces.

### 3.3 Command-line inputs to the controller

Typical full run, from the `exp1/` directory (working directory must contain `run_game.py`):

```bash
python run_batch.py
```

Useful knobs (all optional; defaults match the design above):

| Argument | Default | Meaning |
|----------|---------|---------|
| `--runs` | 5 | Repeats per game per combination |
| `--duration` | 240 | Seconds each game is allowed to run |
| `--models` | cnn tinybert qwen | Which AI jobs |
| `--intensities` | high mid low idle | Request rates |
| `--devices` | gpu cpu | Placement modes for cnn/tinybert |
| `--warmup` | 300 | Seconds to wait after starting the AI before treating it as loaded |
| `--ready-timeout` | 120 | Extra wait until the log contains `Starting <model>` |
| `--baseline-duration` | 240 | Seconds of GPU-temp sampling for the per-combination baseline |
| `--temp-margin` | 2 | °C added to that baseline (then rounded up) |
| `--idle-temp` | 45 | GPU °C to reach before the *next* combination |
| `--poll-interval` | 5 | Seconds between GPU temperature checks |
| `--cooldown-timeout` | 600 | Max wait for either cooldown |
| `--cpu-interval` | 30 | Per-core CPU sample period in milliseconds |
| `--results-dir` | `results` | Where combination folders go |
| `--per-core-dir` | `per_core_result` | Where per-core CSVs go |
| `--ai-script` / `--ai-python` | local script / current interpreter | If the AI environment lives elsewhere |
| `--fresh` | off | Ignore `state.json` and redo everything |
| `--dry-run` | off | Print the plan only |
| `--skip-preflight` | off | Skip the startup sanity checks |

Standalone (not the full sweep):

```bash
python run_game.py xonotic 240
python run_game_batch.py 5 240 --target-temp 45
python run_ai_workload.py cnn 240 high --device gpu
```

---

## 4. What a combination does, in order

For one name such as `cnn_high_gpu`:

1. Create `results/cnn_high_gpu/` (if that folder already exists from a crashed attempt, it is renamed with a `_prev_<timestamp>` suffix so old files are not mixed with new ones).
2. Start `run_ai_workload.py` in the background for a very long duration (24 h). The batch script will stop it with SIGINT. Stdout/stderr go to `ai_workload.log`.
3. Wait `--warmup` seconds, then until the log shows the model has started serving.
4. Sample GPU temperature for `--baseline-duration` seconds; compute `cooldown_target`.
5. Start `cpu_log.py` (must sit next to `run_batch.py`). It writes `per_core_result/cnn_high_gpu.csv`.
6. Run `run_game_batch.py 5 240 --target-temp <cooldown_target>`:
   - iteration 1: xonotic → cooldown → glmark2 → cooldown → luanti → cooldown → supertuxkart → cooldown
   - … repeat until 5 iterations (no cooldown after the very last run).
7. Each `run_game.py` call:
   - uses padding **0** when called from the batch (no extra 10 s idle),
   - starts MangoHud around the game,
   - starts an `nvidia-smi` sidecar logger,
   - kills the game after `--duration` seconds,
   - waits 3 s for MangoHud to flush, then stops `nvidia-smi`.
8. Stop the per-core logger (SIGINT so the CSV is flushed).
9. Move every **new** file that appeared under `mangohud_raw/<game>/` into `results/<combo>/<game>/`.
10. Stop the AI (SIGINT so it prints its throughput/latency summary into the log).
11. Write `meta.json`. Mark the combination completed in `results/state.json`.
12. Cool the GPU to `--idle-temp` (45 °C) before the next combination.

If you interrupt with Ctrl+C, re-run the same command. Completed names in `state.json` are skipped. Failed names can be retried. Partial MangoHud files from the interrupted combination are still collected.

---

## 5. Outputs — everything that is saved

### 5.1 Directory layout after a sweep

```text
exp1/
  mangohud_raw/<game>/          temporary drop folder; emptied into results/ after each combo
  results/
    state.json
    <model>_<intensity>_<device>/
      meta.json
      ai_workload.log
      xonotic/                  MangoHud CSVs + nvidia-smi CSVs for that game
      glmark2/
      luanti/
      supertuxkart/
  per_core_result/
    <model>_<intensity>_<device>.csv
```

`<device>` is `gpu`, `cpu`, or `ollama`. For the two Qwen passes, the folders are separated by the `--results-dir` / `--per-core-dir` choice described in §1.3.

### 5.2 `results/state.json`

Progress file for the whole sweep. Fields:

- `completed` — list of combination names that finished
- `failed` — map of combination name → error string (AI died, never became ready, `run_game_batch.py` non-zero exit, GPU temp unreadable, …)

This is the resume mechanism. It is not per-frame data.

### 5.3 `results/<combo>/meta.json`

One JSON object describing **that** combination. Typical keys:

| Key | Meaning |
|-----|---------|
| `combo` | Name, e.g. `tinybert_mid_cpu` |
| `model`, `intensity`, `device` | The three factors |
| `runs_per_game` | Repeat count (5) |
| `duration_s` | Game length (240) |
| `started`, `finished` | Local ISO timestamps |
| `baseline_avg` | Mean GPU °C during the AI-only baseline window |
| `cooldown_target` | Integer °C used between game runs |
| `per_core_csv` | Path of the per-core log for this combo |
| `cpu_interval_ms` | Sample interval (30) |
| `csv_counts` | Per game, how many non-empty MangoHud CSVs were collected (excludes `*_summary.csv`) |

If `csv_counts` for a game is less than 5, that game produced fewer HUD logs than repeats (crash, HUD failed to flush, empty file).

### 5.4 `results/<combo>/ai_workload.log`

Captured stdout/stderr of `run_ai_workload.py` for the whole time the AI was alive (warmup + baseline + all 20 game runs). Contains load messages, per-request errors (first five), and on SIGINT the **summary**: total / fulfilled / failed requests, throughput, mean latency, min / p50 / p95 / p99 / max latency. This is the record of AI-side performance for the combination.

### 5.5 MangoHud CSV (`results/<combo>/<game>/*.csv`)

Primary **game** trace. MangoHud is configured with:

- `autostart_log=1`
- `log_interval=100` (100 ms)
- `log_duration = duration - 10` (230 s of a 240 s run), so the file is closed **before** the process is killed
- `output_folder = mangohud_raw/<game>` (later moved)
- sensors: `cpu_stats`, `gpu_stats`, `cpu_temp`, `gpu_temp`, `cpu_power`, `gpu_power`, `gpu_core_clock`, `gpu_mem_clock`, `vram`, `ram`, `core_stats`
- `gpu_list=0` unless disabled

Expect one main CSV per successful run, plus often a `*_summary.csv` (MangoHud’s own averages). The collector counts only non-empty files that do **not** end in `_summary.csv` when checking “5 files per game”.

Typical columns (names follow MangoHud’s version): FPS, frame time, CPU load, GPU load, CPU/GPU temperature, CPU/GPU power, core and memory clocks, VRAM, RAM. Use these for frame-time and GPU-side plots.

**Caveat:** on hybrid laptops MangoHud can attach to the iGPU. That is why a second logger exists.

### 5.6 `nvidia_smi_YYYYMMDD_HHMMSS.csv` (same game folders)

Sidecar from:

```text
nvidia-smi --query-gpu=timestamp,utilization.gpu,temperature.gpu,power.draw,clocks.gr,clocks.mem,memory.used
           --format=csv -lms 100
```

One row about every **100 ms**, from NVIDIA’s NVML, independent of MangoHud. Columns:

| Column | Content |
|--------|---------|
| timestamp | Wall-clock time from `nvidia-smi` |
| utilization.gpu | GPU compute utilisation (%) |
| temperature.gpu | GPU °C |
| power.draw | Board power |
| clocks.gr | Graphics clock |
| clocks.mem | Memory clock |
| memory.used | VRAM used |

One file per game run. Align with MangoHud using timestamps.

### 5.7 `per_core_result/<combo>.csv`

Written by `cpu_log.py` for the whole game-batch window of that combination (all four games × 5 runs, including cooldowns between them). Sampled from `/proc/stat` every **30 ms** by default.

| Column | Content |
|--------|---------|
| `timestamp` | Unix epoch seconds |
| `elapsed_s` | Seconds since this logger started |
| `cpu0`, `cpu1`, … | Busy percentage of that logical core over the last interval (100 × busy_ticks / total_ticks; idle includes iowait) |

If a file already exists from a previous attempt, it is renamed `*_prev_YYYYMMDD_HHMMSS.csv` instead of overwritten.

`run_batch.py` **requires** `cpu_log.py` in the **same folder** as itself.

### 5.8 `mangohud_raw/`

Not a result folder. It is a staging area. After each combination, newly created files are moved out. Leftovers mean a crash happened before collection, or a manual `run_game.py` was used.

### 5.9 CIFAR-10 `data/` next to the AI script

Created on first CNN run (`torchvision` download). Dataset cache, not a benchmark metric.

---

## 6. Scripts in `exp1/` (every file)

### `run_batch.py` — sweep controller

This is the script a third party should run for the published design. It does not render frames or run neural nets itself. It:

- builds the list of combinations (Qwen forced to device `ollama`)
- checks games, MangoHud, `nvidia-smi`, `timeout`, CUDA, Ollama, and `cpu_log.py` (preflight)
- starts/stops the AI job and the CPU logger
- measures baseline GPU temperature and computes `cooldown_target`
- calls `run_game_batch.py`
- moves logs into `results/`
- writes `meta.json` and `state.json`
- cools to 45 °C between combinations

### `run_game_batch.py` — five repeats of all four games

Arguments: `n` (repeats), `t` (seconds), `--target-temp`, `--poll-interval`, `--cooldown-timeout`.

Nested loops: for `i` in 1..n, for each game in `GAMES_CONFIG` order (xonotic, glmark2, luanti, supertuxkart), call `run_game.py <game> t --padding 0`, then wait for GPU ≤ `--target-temp` unless this was the last run.

When called from `run_batch.py`, `--target-temp` is the **dynamic** AI-baseline + 2 °C value, not 45.

### `run_game.py` — one game, one timed run

Holds `GAMES_CONFIG` (the four games and their command lines). Sets PRIME offload, builds `MANGOHUD_CONFIG`, launches `mangohud` + game in a new process session, starts `nvidia-smi -lms 100` into `mangohud_raw/<game>/nvidia_smi_<timestamp>.csv`, waits `duration` seconds, sends SIGINT then SIGKILL to the game's process group, sleeps 3 s, stops `nvidia-smi`.

`--padding` default 10 s before and after is only for manual use. The batch always passes `--padding 0` so the only waits between games are the temperature cooldowns.

`--pci-dev` can pin MangoHud to a PCI id if the HUD is logging the wrong GPU.

### `run_ai_workload.py` — background AI job

Loads one of cnn / tinybert / qwen, then issues requests until `duration` or SIGINT. Intensity is the sleep table in §1.4. `--device` selects PyTorch CPU vs CUDA for cnn and tinybert; it has no effect on where Qwen runs (see §1.3). On exit it prints a latency and throughput summary to stdout, which `run_batch.py` captures as `ai_workload.log`.

### `cpu_log.py` — per-core utilisation

Lives next to `run_batch.py`. Reads `/proc/stat` and writes `timestamp,elapsed_s,cpu0,...`. SIGINT flushes and exits. Sampling interval comes from `--cpu-interval` (30 ms).

---

## 7. How to read the modes in the data

For a fixed game, intensity, and AI model:

| Compare | Folder A | Folder B | Interpretation |
|---------|----------|----------|----------------|
| Shared GPU vs AI-on-CPU (cnn, tinybert) | `*_gpu` | `*_cpu` | Effect of putting inference on the same GPU as the game versus leaving the GPU to the game |
| Shared GPU vs AI-on-CPU (qwen) | Ollama-on-GPU pass folders | Ollama-on-CPU pass folders | Same comparison, using the two separate Qwen passes from §1.3 |
| Intensity | `*_high_*` vs `*_mid_*` vs `*_low_*` vs `*_idle_*` | same | Effect of how often inference runs; `idle` is “model resident, no traffic” |
| Game | subfolders `xonotic`, `glmark2`, … | same combo | Different rendering/CPU behaviour under the same AI |

Use MangoHud for FPS/frame time, `nvidia_smi_*.csv` for discrete-GPU truth, `per_core_result/*.csv` for CPU-core busy %, and `ai_workload.log` for inference latency and throughput.

---

## 8. Practical notes for a third party repeating the run

1. Work in `exp1/` so relative output paths (`mangohud_raw`, `results`) stay together.
2. Change the Xonotic and Luanti paths in `run_game.py` if your install locations differ.
3. For Qwen, start Ollama in the intended placement (§1.3) **before** `run_batch.py`, and use separate results folders for the GPU and CPU passes.
4. Do a 30 s single game (`python run_game.py glmark2 30`) and confirm both a MangoHud CSV and an `nvidia_smi_*.csv` appear.
5. A full default sweep takes roughly 31 hours before cooldowns (§1.5).

---
---

# Experiment 2 — CPU demand of each game (games run alone)

Experiment 2 asks a single question:

> How much CPU does each game need when it runs **by itself**, with no AI job alongside it?

No AI workload is started in this experiment. Each game is run alone, on the NVIDIA GPU, while the CPU is sampled at several time resolutions. The result is a per-game picture of how many cores' worth of CPU the game uses, how that splits between P-cores and E-cores, and how many threads and processes are behind it. It is the CPU-side baseline that the Experiment 1 co-runs can be read against.

As in Experiment 1, there is **no CPU-core pinning**: the scheduler places threads wherever it wants.

---

## 1. What is being tested

### 1.1 Four games

The same four titles as Experiment 1, launched in the same repeatable, non-interactive modes (resolution, demo file, random input and demo laps are the same). The frame cap and vsync state of each game during the runs are:

| Game | Frame cap | Vsync | Notes |
|------|-----------|-------|-------|
| **Xonotic** | none (uncapped) | off | 1920×1080, plays `demos/rnd_demo.dem` |
| **glmark2** | none (uncapped) | not present | 1920×1080 |
| **Luanti** | 60 fps | off | `--go --random-input` |
| **SuperTuxKart** | 144 fps | on | demo mode, 8 karts, 100 laps, 1920×1080 |

Frame caps matter when reading the results (see §7): a capped game's CPU demand is partly set by the cap, while an uncapped game takes as much CPU as it can use.

The game is forced onto the discrete NVIDIA GPU via PRIME render offload (`__NV_PRIME_RENDER_OFFLOAD=1`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`), so it does not render on the Intel integrated GPU.

### 1.2 What is measured

Four CPU measurements are taken during every run. They overlap on purpose, because each has a different resolution and a different view of the machine.

| Measurement | Source | Resolution | What it sees |
|-------------|--------|------------|--------------|
| **Demand profile** (`demand.csv`) | `demand_profile.py` reading `/proc` | 0.25 s | CPU time of every thread of every process in the game's session, as “cores-worth”, split into P-cores and E-cores |
| **Per-thread log** (`pidstat_raw.txt`) | `pidstat -t -u` | 1 s | Per-thread CPU use of one game process, and which logical CPU each thread was on |
| **Process tree** (`process_tree.txt`) | `ps --forest` in a loop | 10 s | Every process in the game's session with its cumulative CPU seconds |
| **System per-core log** (`cpu_log.csv`) | `cpu_log.py` reading `/proc/stat` | 30 ms | Busy % of every logical CPU, for the whole machine |

`demand_profile.py` and `pidstat` collect essentially the same kind of data (per-thread CPU time). They differ in sampling interval, and in that `demand_profile.py` covers the whole process session while `pidstat` watches a single process.

### 1.3 Repeats

For **each game**, the run is repeated **3 times** (iterations), each lasting **240 seconds**, with **10 seconds** of padding before and after the game. That is 4 games × 3 iterations = **12 runs**, executed game by game:

```text
xonotic iter 1,2,3 → glmark2 iter 1,2,3 → luanti iter 1,2,3 → supertuxkart iter 1,2,3
```

Each run takes roughly 263 s (10 s pre-padding + 240 s game + 3 s flush + 10 s post-padding) plus the cooldown wait, so the whole experiment is a little under an hour before cooldowns.

### 1.4 How this differs from Experiment 1

| | Experiment 1 | Experiment 2 |
|---|--------------|--------------|
| AI job | Running during every game | None |
| Question | Effect of AI on game and GPU/CPU | CPU needed by each game alone |
| Controller | `run_batch.py` | `run_all.py` |
| Game launcher | `run_game.py` | `run_game_cpu_core.py` |
| Cooldown target | Adaptive (AI baseline + 2 °C) | Fixed 50 °C on CPU **and** GPU |
| Cooldown applied | Between runs | Before every run, including the first |
| Padding | 0 (batch passes `--padding 0`) | 10 s before and after |
| Data kept | MangoHud, `nvidia-smi`, per-core CSV, AI log | CPU data only (see §5) |

---

## 2. Temperature control

Before **every** run, `run_all.py` waits until the CPU and the NVIDIA GPU are both at or below the target temperature:

- Target: `--temp-target`, default **50 °C**. The resting temperature used for the runs was 50 °C. A value of `0` disables the wait.
- Polling every **5 s**, with a status line about every 15 s.
- Timeout: `--temp-timeout`, default **600 s**. If the machine has not cooled in time, the run starts anyway and a warning is printed.
- CPU temperature is read from `hwmon` (`coretemp`, `k10temp`, `zenpower` or `cpu_thermal`, preferring “Package id 0”, `Tctl` or `Tdie`), falling back to the `x86_pkg_temp` thermal zone. GPU temperature comes from `nvidia-smi`.
- If one sensor cannot be found, a warning is printed and that sensor is ignored. If neither is found, no wait is done.

---

## 3. Inputs you must provide

### 3.1 Machine and software

- Linux (reads `/proc` and `/sys`)
- NVIDIA driver, with `nvidia-smi` available
- **MangoHud** on `PATH` (the launcher still wraps the game with it)
- **`pidstat`** on `PATH` (from the `sysstat` package)
- `ps` and `bash` (used by the process-tree logger); `stdbuf` is used if present
- Python 3 (standard library only for the four scripts in this experiment)
- The four scripts must sit **in the same folder**: `run_all.py` imports `demand_profile.py`, and `run_game_cpu_core.py` looks for `demand_profile.py` next to itself. `run_all.py` also expects `cpu_log.py` and `run_game_cpu_core.py` there by default.

### 3.2 Game binaries and working directory

`run_game_cpu_core.py` has **its own copy** of `GAMES_CONFIG`; it does not import the one from `run_game.py`. Edit paths in this copy for Experiment 2.

- **Xonotic** is given as a relative path (`../xonotic/xonotic/all`), so `run_all.py` must be started from a folder where that path resolves.
- **Luanti** is still an absolute local path (`/run/media/tanish/...`). On another machine this path would have to be changed.
- glmark2 and SuperTuxKart are expected as system commands.

As in Experiment 1, do not change launch arguments if you want comparable traces.

### 3.3 Command-line inputs

The settings used for the reported runs (240 s per run, 3 iterations, 10 s padding, 50 °C target) are the script defaults apart from the duration:

```bash
python run_all.py 240 -n 3
```

Arguments of `run_all.py`:

| Argument | Default | Meaning |
|----------|---------|---------|
| `duration` (positional) | — | Seconds per run, passed to `run_game_cpu_core.py` |
| `-n`, `--iterations` | 3 | Iterations per game |
| `-g`, `--games` | all four | Which games to run |
| `-o`, `--output` | `cpu_runs` | Output root folder |
| `--cpu-interval` | 30 | `cpu_log.py` interval in ms |
| `--padding` | 10 | Pre/post padding in seconds, passed to the game launcher |
| `--pidstat-interval` | 1 | `pidstat` interval in seconds |
| `--temp-target` | 50 | °C both CPU and GPU must reach before each run (0 disables) |
| `--temp-timeout` | 600 | Max seconds to wait for cooldown per run (0 = no limit) |
| `--cpu-script` / `--game-script` | scripts in the same folder | Paths to `cpu_log.py` / `run_game_cpu_core.py` |

Standalone pieces:

```bash
python run_game_cpu_core.py glmark2 60                  # one game, outputs in mangohud_raw/glmark2/
python demand_profile.py --summary demand.csv           # p50/p95/max summary of an existing demand.csv
python cpu_log.py -i 30 -d 60 -o cpu_log.csv            # per-core logger by itself
```

`run_game_cpu_core.py` also accepts `--demand-dt` (default 0.25 s), `--pci-dev`, `--gpu-list` and `--debug`. `run_all.py` does not pass these on, so the sampler always runs at 0.25 s in a normal experiment.

---

## 4. What one run does, in order

For one game and iteration, such as `xonotic` iteration 1:

1. **Cool down** until CPU and GPU are ≤ `--temp-target` (§2).
2. Delete `mangohud_raw/<game>/` in the working directory, so no stale files remain from an earlier run.
3. Start `cpu_log.py` in the background, writing `<run_dir>/cpu_log.csv`.
4. Run `run_game_cpu_core.py <game> <duration> --padding 10 --pidstat-interval 1`, which does the following:
   1. Wait the 10 s pre-run padding.
   2. Launch `mangohud` + the game in a **new process session** (so the whole tree can be sampled and killed together).
   3. Start `demand_profile.py` on that session (0.25 s sampling).
   4. Start the process-tree logger (`ps --forest` every 10 s).
   5. Start the `nvidia-smi` sidecar logger (100 ms).
   6. Find the right process for `pidstat`. Some games start through a wrapper shell script (Xonotic's `all`) that sits idle while the real game runs as its child. A wrapper has one thread and a real game has several, so the script waits (up to 60 s) for a multi-threaded process and uses that one. Then start `pidstat -t -u -p <pid> 1`.
   7. Wait `duration` seconds, then send SIGINT to the game's process group, and SIGKILL if it has not exited after 10 s.
   8. Wait 3 s for MangoHud to flush, then stop `nvidia-smi`, `pidstat`, `demand_profile.py` and the process-tree logger.
   9. Convert `pidstat_raw.txt` into `cpu_cores_used.csv` (distinct cores per timestamp) and print a short summary.
   10. Wait the 10 s post-run padding.
5. Stop `cpu_log.py` with SIGINT so it flushes its CSV.
6. **Keep three files** from `mangohud_raw/<game>/` (`pidstat_raw.txt`, `demand.csv`, `process_tree.txt`) by copying them into the run folder, then delete everything else in `mangohud_raw/<game>/` (§5.3).
7. Print the demand summary and the process-tree check (§5.7).
8. Wait 2 s, then move on to the next iteration or game.

Because `cpu_log.py` is started before the game launcher and stopped after it, `cpu_log.csv` also contains the padding periods; `demand.csv` and `process_tree.txt` cover only the game's own lifetime.

---

## 5. Outputs — everything that is saved

### 5.1 Directory layout after a run

```text
cpu_runs/
  <game>/                        xonotic, glmark2, luanti, supertuxkart
    iter_01/
      cpu_log.csv
      pidstat_raw.txt
      demand.csv
      process_tree.txt
    iter_02/
    iter_03/
```

### 5.2 `demand.csv` — the main result

Written by `demand_profile.py` every 0.25 s. It samples `utime + stime` of **every thread of every process in the game's session** from `/proc`, so child processes and helper threads are included without needing any PIDs. *Cores-worth* means CPU seconds used per wall-clock second.

| Column | Content |
|--------|---------|
| `epoch` | Wall-clock Unix seconds (same clock as `cpu_log.csv` `timestamp`) |
| `t` | Seconds since the sampler started |
| `total_cores` | Cores-worth used by the whole session |
| `p_cores` | Part of that on P-core logical CPUs (0–11) |
| `e_cores` | Part of that on E-core logical CPUs (12–15) |
| `busiest_thread_cores` | Cores-worth of the single busiest thread in that interval |
| `n_threads` | Threads seen |
| `n_procs` | Processes in the session |

The process list is refreshed every 1 s to pick up child processes started later. The rate is computed from the real elapsed time of each interval, not the nominal 0.25 s. `python demand_profile.py --summary demand.csv` (and `run_all.py` after each run) prints the p50 / p95 / max of total cores-worth, and the p95 of P-cores, E-cores and the busiest thread.

### 5.3 What is deleted

Everything else the game launcher produces is deleted by `run_all.py`: the MangoHud CSVs, the `nvidia_smi.csv`, `pidstat_err.txt`, and the `cpu_cores_used.csv` that was just built from the pidstat log. So **Experiment 2 has no FPS, GPU or power data**, and the pidstat log is kept only in its raw form.

### 5.4 `pidstat_raw.txt`

Raw output of `pidstat -t -u -p <pid> 1` (per-thread CPU use of one process, 1 s interval, 24 h ISO timestamps, `LC_ALL=C`). Each row gives a thread's CPU use and the logical CPU it was running on at that moment. It is converted to a CSV afterwards in a separate Jupyter step (not part of this folder). That conversion follows the same idea as the `parse_pidstat` function in `run_game_cpu_core.py`: for each timestamp, the sorted list of **distinct CPUs** the game's threads were on (`timestamp, num_cores, cores`).

### 5.5 `process_tree.txt`

Every 10 s: a time stamp, then `ps --forest -o pid,ppid,nlwp,cputimes,comm` for the game's session (process ID, parent ID, number of threads, cumulative CPU seconds, command). It shows which processes belong to the game and how much CPU each has used, so child processes cannot be missed silently.

### 5.6 `cpu_log.csv`

Written by `cpu_log.py` (the same script as in Experiment 1) every **30 ms** from `/proc/stat`, for the **whole machine**:

| Column | Content |
|--------|---------|
| `timestamp` | Unix epoch seconds |
| `elapsed_s` | Seconds since the logger started |
| `cpu0` … `cpu15` | Busy % of that logical CPU over the last interval (100 × busy ticks / total ticks; idle includes iowait) |

### 5.7 Console summary printed after every run

- **Demand summary:** the p50/p95/max lines from §5.2.
- **Process-tree check:** the top five processes in the session by CPU seconds, with their share. If more than one process uses at least 5 % of the total, a warning says that `pidstat` watches only one of them. If one process does essentially all the work, the script reports that `pidstat` is measuring the right thing. This check does not affect `demand.csv`, which covers the whole session either way.

---

## 6. Scripts used in Experiment 2

### `run_all.py` — controller

The main operator. For each game and iteration it waits for the temperature target, starts `cpu_log.py`, runs the game launcher, stops `cpu_log.py`, keeps the three files described above, deletes the rest, and prints the summaries. It converts SIGTERM and SIGHUP into a normal exit so `cpu_log.py` is still stopped if the terminal is closed.

### `run_game_cpu_core.py` — one game, one timed run with CPU samplers

A variant of `run_game.py`. It sets up MangoHud and PRIME offload, launches the game in a new session, starts the four samplers (`demand_profile.py`, the process-tree logger, `nvidia-smi`, `pidstat`), stops the game after the duration (SIGINT, then SIGKILL), and does the final pidstat parsing. MangoHud and `nvidia-smi` still run here, but their output is discarded by `run_all.py`.

### `demand_profile.py` — cores-worth sampler

Samples a whole process session every 0.25 s and writes `demand.csv`. It can also print a summary of an existing file with `--summary`. P-core and E-core CPU lists are set by `--p-cpus` (default `0-11`) and `--e-cpus` (default `12-15`); these match the test laptop (Appendix A). `run_game_cpu_core.py` launches it without those options, so on a different CPU the defaults inside `demand_profile.py` would have to be edited.

### `cpu_log.py` — per-core utilisation

Unchanged from Experiment 1: per-core busy % from `/proc/stat`, every 30 ms by default.

---

## 7. How to read the data

| Question | Where to look |
|----------|---------------|
| How much CPU does the game need? | `demand.csv` `total_cores`: p50 for the typical load, p95 for near-peak demand, max for the worst moment |
| Does it run on P-cores or E-cores? | `p_cores` vs `e_cores` |
| Is it limited by one thread? | `busiest_thread_cores`. A value near 1.0 means one thread is saturating a logical CPU |
| How many threads and processes are involved? | `n_threads`, `n_procs`, and `process_tree.txt` |
| Which logical CPUs did the game's threads land on? | `pidstat_raw.txt` (after conversion) |
| What was the whole machine doing, per core, at fine resolution? | `cpu_log.csv`, aligned with `demand.csv` via `timestamp` / `epoch` |

Points to keep in mind when interpreting results:

- **Cores-worth counts logical CPUs.** The P-cores have two threads each (SMT) that share one physical core, so a cores-worth of 2.0 spread over two sibling threads is not the same as two full physical cores.
- **Frame caps shape the result.** Luanti (60 fps) and SuperTuxKart (144 fps, vsync on) are limited by their caps, while Xonotic and glmark2 are uncapped and will use whatever CPU they can.
- **P/E attribution is approximate.** Each thread's CPU time in an interval is credited to the core type it was on at the **end** of that interval.
- **Sampling limits.** Kernel CPU time ticks are 10 ms, so intervals below about 0.2 s are noisy. Processes that start and exit between two 1 s session scans are not seen.
- **Compare with Experiment 1 with care.** Experiment 2 has no AI job running, uses padding, and holds the CPU in the `performance` governor (Appendix A).

---

## 8. Practical notes for a third party repeating the run

1. Put `run_all.py`, `run_game_cpu_core.py`, `demand_profile.py` and `cpu_log.py` in one folder, and start `run_all.py` from a folder where the game paths in `run_game_cpu_core.py` resolve.
2. Edit the game paths in `run_game_cpu_core.py` (not `run_game.py`) if your install locations differ. Luanti's path is hardcoded to the test machine.
3. **Use a working directory that does not hold Experiment 1 staging files.** `run_all.py` deletes `./mangohud_raw/<game>/` before and after every run, and Experiment 1 uses the same folder name for its temporary files.
4. **Re-running overwrites.** Writing into an existing `cpu_runs/<game>/iter_XX/` replaces its files. Rename or move old results first (unlike Experiment 1, nothing is renamed automatically).
5. Check your CPU layout with `lscpu -e=CPU,CORE,SOCKET,MAXMHZ,MINMHZ` and edit the P/E defaults in `demand_profile.py` if it differs from Appendix A.
6. Install `sysstat` so that `pidstat` exists.
7. Do a short test first, for example `python run_all.py 60 -n 1 -g glmark2`, and confirm that all four files appear in `cpu_runs/glmark2/iter_01/`.
8. Keep the machine in the same state for all runs (power source, power mode, nothing else running). The setup used is listed in Appendix A.

---

# Experiment 3 — Pinned co-run of games and AI workloads (disjoint CPU cores)

Experiment 3 asks a single question:

> If the game and the AI job are each confined to their own set of CPU cores with `taskset`, does the game get back the performance it lost to the AI in Experiment 1?

The setup is the Experiment 1 co-run (a game and an AI inference job running on the same machine at the same time) with one deliberate change: the **game is pinned to a small set of logical CPUs and the AI job is pinned to the remaining ones**, so the two can never be scheduled onto the same logical CPU. The games, their launch arguments, the 240 s window and the AI request loop are the same as before. What Experiment 3 adds is a set of checks that prove the split really held in every run (partition validation before launch, and read-backs of the real affinity masks afterwards).

Pinning is the intended difference, but it is not the only one. Read §1.7 and §7 before comparing numbers with Experiment 1.

---

## 1. What is being tested

### 1.1 Four games

The same four titles as Experiments 1 and 2 (Xonotic, glmark2, Luanti, SuperTuxKart), with the same resolution, demo file, random input and demo laps. The game is still forced onto the NVIDIA GPU by PRIME render offload (`__NV_PRIME_RENDER_OFFLOAD=1`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`), so only the **CPU** side of the placement is new.

Pinning works like this: the launched command is `mangohud taskset -c <game cores> <game command>`. `taskset` execs the game, so the game and every thread and child process it creates inherit the mask. The game CPUs are **not** hardcoded in `GAMES_CONFIG`. They come from `--game-cores` (or, if that is absent, from `DEFAULT_GAME_CORES` in `run_game.py`), so the same launcher serves any partition. `--game-cores unpinned` removes the `taskset` entirely.

**This `run_game.py` is not the Experiment 1 launcher.** It adds the partition check, the affinity read-back and the telemetry pin (§6). Keep the two in separate folders.

The Xonotic and Luanti executables are absolute paths to a local disk inside `run_game.py`. On another machine they have to be edited. glmark2 and SuperTuxKart are expected as system commands.

### 1.2 Three AI workloads, two controllers

| Name | Controller | How the request is defined |
| --- | --- | --- |
| **cnn** | `run_task.py` | CIFAR-10 ResNet20, one forward pass of a random batch (as Experiment 1 §1.2) |
| **tinybert** | `run_task.py` | `prajjwal1/bert-tiny` on SST-2, one forward pass of up to 32 random sentences (as Experiment 1 §1.2) |
| **qwen** | `run_task_ollama.py` | One non-streaming HTTP generate call to a local Ollama server (`qwen:0.5b`, Dolly-15k prompts, 120 s timeout) |

The AI script is `run_ai_workload.py`. For cnn and tinybert it is the process that runs the model, so `run_task.py` starts it under `taskset`. For qwen it only sends HTTP requests; the inference runs inside the Ollama server. Pinning the request script therefore does nothing for the model, so **you** start the Ollama server pinned (§1.3) and `run_task_ollama.py` only checks it and never changes it.

### 1.3 Placement modes

| Mode | cnn and tinybert | qwen |
| --- | --- | --- |
| **gpu** | `--device gpu`: model on the NVIDIA GPU | Ollama started normally (default) |
| **cpu** | `--device cpu`: model on the CPU | Ollama started with the GPU hidden (below) |

The AI process is pinned in **both** placements. With the model on the GPU, the CPU-side work (request loop, tokenisation, kernel launches) is still confined to the AI's cores; with the model on the CPU, the inference itself is.

For qwen, start the server pinned to the AI cores of the batch you are about to run:

```bash
taskset -c 4-15 ollama serve                                # xonotic, Ollama on GPU
taskset -c 2-15 ollama serve                                # other games, Ollama on GPU
CUDA_VISIBLE_DEVICES='' taskset -c 2-15 ollama serve        # other games, Ollama on CPU (use 4-15 for xonotic)
```

The script reads `/api/ps` and fails the run if qwen is not on the requested device (`gpu`, `cpu`, or `split` when only part of the model is in VRAM). Leave each server running for a whole batch; restarting it mid-batch would mix conditions in the same folders.

### 1.4 CPU partitions

| Game | Game CPUs | AI CPUs | AI thread cap |
| --- | --- | --- | --- |
| Xonotic | `0-2` | `4-15` | 12 |
| glmark2, Luanti, SuperTuxKart | `0-1` | `2-15` | 14 |

CPU 3 is left idle with Xonotic on purpose: it is the SMT sibling of CPU 2, and the rule is that **a physical core belongs to one side only**.

The partition is validated by `cpu_partition.py` before anything is launched, and again by `run_game.py` before the game starts. The check uses the real sibling pairs from `/sys/devices/system/cpu/cpuN/topology/thread_siblings_list` (not an assumed layout) and fails if any of these hold:

- a mask contains a CPU that does not exist on the machine;
- the game and the AI (or the telemetry CPU) share a logical CPU;
- a physical core with two or more threads is split between the game and the AI (or the telemetry CPU).

The defaults can be overridden: `--game-cores` and `--ai-cores` apply to all selected games, `--ai-cores auto` means every CPU that does not share a physical core with the game, and `--game-cores-xonotic`, `--game-cores-default`, `--ai-cores-xonotic`, `--ai-cores-default` set the per-game defaults. The same check can be run by hand: `./cpu_partition.py 0-2 4-15`.

**Thread cap.** A pinned process that starts more threads than it has CPUs would oversubscribe its own partition and look slower for reasons unrelated to interference. So the AI is capped to the partition size: `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `AI_NUM_THREADS` are set to that size, `run_ai_workload.py` is started with `--threads N` (torch intra-op threads and the Rust tokenizer pool), and for qwen the request carries `num_thread = N` so Ollama sizes its pool from the mask and not from the machine.

`taskset` fixes which CPUs a process **may** use. It does not decide where its threads run: the scheduler still chooses among the allowed CPUs.

### 1.5 Request intensity

The same sleep-after-each-request settings as Experiment 1 §1.4: `high` 0.05 s, `mid` 0.5 s, `low` 2.0 s, `idle` (no requests, the process sleeps 1 s at a time with the model resident).

### 1.6 Repeats and size

For **each** combination and **each** game, the run is repeated **5 times** with a **240 s** game window. A **new AI process is started for every run**.

| Controller | Combinations | Runs | Estimated time |
| --- | --- | --- | --- |
| `run_task.py` (cnn, tinybert) | 2 models × 4 intensities × 2 devices = 16 | 16 × 4 games × 5 = **320** | at least **33 h 46 m** |
| `run_task_ollama.py` (qwen), one device pass | 4 intensities | 4 × 4 games × 5 = **80** | at least **8 h 26 m** |

Both scripts estimate each run as 120 s (AI alone) + 240 s (game) + 20 s overhead = 380 s. The estimates exclude model loading and cooldown waits. Qwen has to be split into batches by AI core set (Xonotic: 20 runs, about 2 h 06 m; the other three games: 60 runs, about 6 h 20 m), and the CPU placement is a second pass of the same size.

### 1.7 How this differs from Experiment 1

|  | Experiment 1 | Experiment 3 |
| --- | --- | --- |
| Game's CPUs | any | 2 or 3 logical CPUs |
| AI's CPUs | any | the rest, minus any sibling of the game's cores |
| AI thread cap | none | equal to the AI partition size |
| AI lifecycle | one process per combination, 20 game runs back to back | a fresh process per run |
| Before the game starts | AI at thermal steady state | cooled to 50 °C, then 120 s of AI alone |
| Cooldown | adaptive target, GPU only, AI left running | fixed 50 °C on GPU **and** CPU, nothing running |
| Per-core CPU log | whole combination | game window only |
| Qwen | one sweep controlled by `run_batch.py` | separate controller, pinned server verified |
| Verification | none | partition check, game and AI affinity read-back, Ollama affinity and device check |
| Telemetry (`cpu_log.py`, `nvidia-smi`) | unpinned | optionally pinned to one CPU |

---

## 2. Temperature control

Before **every** run, with nothing of ours running, the controller waits until the GPU and the CPU are both at or below the target:

| Setting | Default |
| --- | --- |
| `--gpu-target` | 50 °C |
| `--cpu-target` | 50 °C |
| `--cooldown-timeout` | 600 s |
| `--poll-interval` | 5 s |

- GPU temperature comes from `nvidia-smi`. CPU temperature comes from the `k10temp` or `coretemp` sensor in `/sys/class/hwmon` (preferring `Tctl`, `Tdie` or `Package id 0`, otherwise `temp1_input`).
- `--no-cpu-wait` cools down on GPU temperature only.
- If the target is not reached within the timeout, the run is **still performed** but is flagged `started_hot` in `meta.json`, and it is listed in `state.json` and in the final summary.
- The starting temperatures are stored in `meta.json` (`start_gpu_temp`, `start_cpu_temp`).

Unlike Experiment 1, the target is a fixed 50 °C on both devices, not the AI's own steady-state temperature plus a margin. After the cooldown, the AI runs alone for 120 s before the game starts, so the AI itself is warm at game start but the starting point is the same every time.

---

## 3. Inputs you must provide

### 3.1 Machine and software

- Linux (the per-core logger and affinity checks read `/proc` and `/sys`)
- NVIDIA driver with `nvidia-smi`
- **MangoHud** and **`taskset`** (util-linux) on `PATH`; `pgrep` is used to read thread affinities, and `pkill` to stop the game cleanly (with a fallback if it is missing)
- Python with `torch`, `torchvision`, `transformers`, `datasets` and `requests` for the AI script. If it lives in a virtual environment, point `--ai-python` at it.
- CUDA visible to that Python if any `gpu` combination is run. Otherwise the AI script would silently fall back to the CPU, so the preflight check refuses to start.
- A CPU temperature sensor (`k10temp`/`coretemp`), unless `--no-cpu-wait` is used
- For qwen: Ollama on `http://localhost:11434` with `qwen:0.5b` pulled, started pinned as in §1.3

### 3.2 Files and working directory

Put these in **one folder** and run the controllers from it: `run_task.py`, `run_task_ollama.py`, `run_game.py`, `cpu_partition.py`, `cpu_log.py`.

- The controllers import `GAMES_CONFIG`, `DEFAULT_GAME_CORES` and the MangoHud staging folder (`./mangohud_raw`, relative to the working directory) from `run_game.py`, and they call it by relative name, so the working directory must contain it.
- `run_ai_workload.py` is found through `--ai-script` and is run from its own folder, so the CIFAR-10 `data/` directory stays next to it.
- Use a folder that does not hold Experiment 1 or Experiment 2 staging files, because they use the same `mangohud_raw` name.

### 3.3 Command-line inputs

Full cnn and tinybert sweep:

```bash
python run_task.py
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--runs` | 5 | Runs per game per combination |
| `--duration` | 240 | Game window in seconds |
| `--models` | cnn tinybert | AI models (`run_task.py` only) |
| `--intensities` | high mid low idle | Request intensities |
| `--devices` | gpu cpu (qwen: gpu) | Placement. For qwen it must match how you started Ollama |
| `--games` | all four (qwen: games sharing one AI core set per batch) | Games to run |
| `--game-cores` | per-game default | Game CPUs for all selected games, or `unpinned` |
| `--ai-cores` | per-game default | AI CPUs for all selected games, or `auto` |
| `--game-cores-xonotic` / `--game-cores-default` | `0-2` / `0-1` | Per-game game CPUs |
| `--ai-cores-xonotic` / `--ai-cores-default` | `4-15` / `2-15` | Per-game AI CPUs |
| `--telemetry-cpu` | off | Pin `cpu_log.py` and the `nvidia-smi` sidecar to this one CPU |
| `--gpu-target` / `--cpu-target` | 50 / 50 | °C to reach before each run |
| `--no-cpu-wait` | off | Cool down on GPU temperature only |
| `--cooldown-timeout` | 600 | Max seconds for the cooldown |
| `--poll-interval` | 5 | Seconds between temperature checks |
| `--ai-solo-wait` | 120 | Seconds the AI runs alone before the game starts |
| `--ready-timeout` | 300 | Max seconds for the AI model to load |
| `--ai-script` / `--ai-python` | `run_ai_workload.py` / current Python | AI script and interpreter |
| `--results-dir` | `result_task` (qwen: `result_task_ollama`) | Where results are written |
| `--cpu-interval` | 30 | Per-core sampling interval in ms |
| `--fresh` / `--dry-run` / `--skip-preflight` | off | Redo everything / print the plan only / skip startup checks |
| `--no-check-affinity` / `--no-verify-device` | off | qwen only: skip the Ollama affinity check / the `/api/ps` device check |

Examples:

```bash
python run_task.py --runs 3 --models cnn --intensities high mid --devices gpu
python run_task.py --ai-cores auto --telemetry-cpu 15
python run_task_ollama.py --games xonotic                     # server started with taskset -c 4-15
python run_task_ollama.py --games glmark2 luanti supertuxkart  # server restarted with taskset -c 2-15
```

**Preflight.** Before the first run, `run_task.py` checks that `run_game.py`, `cpu_log.py`, `/proc/stat` and the AI script exist; that `nvidia-smi`, `mangohud` and `taskset` are on `PATH`; that the GPU (and CPU) temperature can be read; that every game executable exists; that the partition is valid for every selected game; and that PyTorch sees CUDA when `gpu` combinations are requested. `run_task_ollama.py` does the same for its games and also checks that Ollama is reachable, that `qwen:0.5b` is pulled, that the server's affinity matches the first game's AI cores, that all selected games use the **same** AI cores (otherwise it asks for separate batches), and that the AI interpreter can import its modules.

---

## 4. What one run does, in order

For one combination, one game and one run number `k` (cnn and tinybert):

1. Create `result_task/<combo>/<game>/run_<k>/`. If it already exists from an interrupted attempt, rename it with a `_prev_<timestamp>` suffix so files are not mixed.
2. **Cool down** until GPU and CPU are ≤ 50 °C (§2) and record the starting temperatures.
3. Start the AI under `taskset` on the AI cores, with the thread cap, writing `ai_workload.log` and `ai_requests.csv`.
4. **Wait until the model has loaded**, meaning `Starting <model>` appears in the AI log (up to `--ready-timeout`). The run fails if the AI exits first.
5. Save the AI's real affinity (`taskset -a -c -p`, every thread) to `ai_affinity.txt`.
6. **AI runs alone** for `--ai-solo-wait` (120 s), so the game always starts against an AI that is already running.
7. Start `cpu_log.py` (30 ms), writing `cpu_percore.csv`.
8. Run `run_game.py <game> 240 --padding 0 --game-cores <..> --ai-cores <..>`, which:
   1. checks the partition again and exits with code 2 if it is not clean;
   2. starts MangoHud, the pinned game and the `nvidia-smi` sidecar (100 ms);
   3. after 5 s, reads the allowed CPUs of **every thread** of every process in the game's session and writes them to `game_affinity_<timestamp>.txt` with a status of `OK`, `MISMATCH` or `NO THREADS FOUND`;
   4. stops the game after 240 s (SIGINT, then SIGKILL after 10 s), waits 3 s for the MangoHud flush, and stops `nvidia-smi`.

   MangoHud logs for the first 230 s so its CSV is closed before the game is killed. Meanwhile the controller checks every 2 s that the AI is still alive.
9. Stop `cpu_log.py` with SIGINT, so the per-core log covers only the game window.
10. Move the new files from `mangohud_raw/<game>/` into the run folder.
11. Stop the AI with SIGINT, so it prints its throughput and latency summary into `ai_workload.log`.
12. Write `meta.json` and record the run in `state.json`.

A run counts as failed if the AI exits early or never becomes ready, `cpu_log.py` does not start, `run_game.py` exits non-zero (including a bad partition), or no MangoHud CSV was produced. Partial data from a failed run is kept, the failure is recorded, and the sweep moves on. An affinity `MISMATCH` does **not** fail the run: it prints a warning and is written to `game_affinity_<timestamp>.txt`, so check that file before using the run.

**Qwen differences (`run_task_ollama.py`).** The first step is to check that every Ollama process (`ollama` and `ollama_llama_server`) is allowed on exactly the AI cores, **before** the cooldown, so a wrong server fails fast. The qwen client (not the server) is started under `taskset` on the AI cores. After the model has loaded, the Ollama affinity is checked again, because the runner process only exists once the model is loaded, and `/api/ps` is read to confirm the device (skipped for `idle`, where no requests are sent). The rest is identical.

If you interrupt with Ctrl+C, run the same command again: completed runs are skipped and failed ones are retried.

---

## 5. Outputs — everything that is saved

### 5.1 Directory layout

```text
result_task/                           (qwen: result_task_ollama/)
  state.json
  <model>_<intensity>_<device>/        e.g. cnn_high_gpu, qwen_mid_cpu
    <game>/                            xonotic, glmark2, luanti, supertuxkart
      run_1/ … run_5/
        <mangohud>.csv                 game trace (FPS, frame time, CPU/GPU sensors)
        <mangohud>_summary.csv         MangoHud's own averages
        nvidia_smi_<timestamp>.csv     GPU trace (100 ms)
        game_affinity_<timestamp>.txt  allowed CPUs of every game thread
        ai_workload.log                AI stdout, with the final summary
        ai_requests.csv                one row per AI request
        ai_affinity.txt                allowed CPUs of every AI thread
        cpu_percore.csv                per-core busy %, game window only
        meta.json                      description of the run
```

### 5.2 `state.json`

Progress file: `completed` (run keys that finished), `failed` (run key → error) and `hot` (runs that started above the cooldown target). It is the resume mechanism.

### 5.3 `meta.json`

| Key | Meaning |
| --- | --- |
| `combo`, `model`, `intensity`, `device`, `game`, `run` | The factors |
| `duration_s` | Game window (240) |
| `status` | `ok`, `failed`, `interrupted` or `incomplete` |
| `started`, `finished` | Local ISO timestamps |
| `started_hot`, `start_gpu_temp`, `start_cpu_temp`, `gpu_target`, `cpu_target` | Cooldown outcome |
| `game_cores`, `ai_cores`, `ai_threads` | The two masks and the AI thread cap |
| `telemetry_cpu` | CPU the loggers were pinned to, or null |
| `per_core_csv`, `cpu_interval_ms` | The per-core log and its interval |
| `game_started_epoch`, `game_finished_epoch` | Unix times of the game window, for aligning logs |
| `mangohud_csvs` | Number of MangoHud CSVs collected (summary, `nvidia_smi` and affinity files excluded) |
| `ai_requests` | `total_requests`, `fulfilled`, `failed`, parsed from the AI log |
| `ollama_model`, `ollama_affinity_pre`, `ollama_affinity`, `ollama_device_actual` | qwen only: the server checks and what they found |
| `error` | Set if the run failed |

### 5.4 MangoHud and `nvidia-smi` files

Same configuration and columns as Experiment 1 §5.5 and §5.6: MangoHud at 100 ms with `log_duration` 230 s, and `nvidia-smi` every 100 ms. These are the game-side results: frame times and FPS from MangoHud, and the discrete GPU's utilisation, temperature, power, clocks and memory from `nvidia-smi`.

### 5.5 `ai_workload.log` and `ai_requests.csv`

The log holds the load messages, the thread-cap line, up to five request errors, and the final summary (total, fulfilled and failed requests, throughput, mean latency, min/p50/p95/p99/max). `ai_requests.csv` has one row per request:

| Column | Content |
| --- | --- |
| `request_id` | 1, 2, 3 … |
| `epoch_s` | Wall-clock start of the request, aligned with the other logs |
| `elapsed_s` | Seconds since the AI loop started |
| `latency_ms` | Request latency |
| `ok` | 1 if it succeeded |
| `tokens` | Tokens generated (qwen only; empty for cnn and tinybert) |

The AI runs for the whole run, so these files include the 120 s AI-alone period. Use `game_started_epoch` and `game_finished_epoch` to select the game window.

### 5.6 `game_affinity_<timestamp>.txt` and `ai_affinity.txt`

These two files are the proof that each process stayed on its own side. `game_affinity_*.txt` starts with the expected mask, the thread count, the distinct masks and a status line, then lists `pid tid comm cpus_allowed_list` for every game thread. `ai_affinity.txt` is the raw `taskset -a -c -p` output for the AI process (for qwen, the client only; the server is checked separately and stored in `meta.json`). Every thread should show exactly its partition.

### 5.6a `cpu_percore.csv`

Written by `cpu_log.py` at 30 ms, starting just before the game and stopping right after it.

| Column | Content |
| --- | --- |
| `timestamp` | Unix epoch seconds |
| `elapsed_s` | Seconds since the logger started |
| `cpu0` … `cpu15` | Busy % of each logical CPU over the last interval |

This is the behavioural check that the partition held: the game's load should sit on its CPUs and the AI's load on the AI's CPUs.

---

## 6. Scripts used in Experiment 3

### `run_task.py` — controller for cnn and tinybert

Builds the list of (model × intensity × device × game × run) items, runs the preflight, and performs each run as in §4. It keeps progress in `state.json`, marks runs that started hot, and prints a summary of completed, hot and failed runs at the end. It also holds the helpers `run_task_ollama.py` reuses (cooldown, AI start and stop, CPU logging, game launch, result collection).

### `run_task_ollama.py` — controller for qwen

The same flow for the Ollama workload, with the additional server checks (affinity, device, model pulled) and the one-AI-core-set-per-batch rule. It never starts, stops or re-pins Ollama.

### `run_game.py` — one game, one timed run (pinned version)

Holds `GAMES_CONFIG` and `DEFAULT_GAME_CORES`. It validates the partition, launches `mangohud` + `taskset` + game in a new process session, runs the `nvidia-smi` sidecar, writes the game affinity read-back, and stops the game after the duration. `--telemetry-cpu` pins the `nvidia-smi` sidecar to one CPU.

### `cpu_partition.py` — partition helper

Parses and formats CPU lists, reads the real SMT sibling groups from sysfs, validates a partition, and computes `auto` AI cores. It can also be run directly as a standalone check.

### `run_ai_workload.py` — the AI job

Loads one of cnn, tinybert or qwen and issues requests at the chosen intensity until its duration ends or it receives SIGINT. It adds `--threads` (or `$AI_NUM_THREADS`) for the thread cap and `--log-csv` for the per-request log.

### `cpu_log.py` — per-core utilisation

Unchanged: per-core busy % from `/proc/stat` every 30 ms.

---

## 7. How to read the data

| Question | Where to look |
| --- | --- |
| Did the partition hold? | `game_affinity_*.txt`, `ai_affinity.txt`, `meta.json` (Ollama checks), then `cpu_percore.csv` |
| Did the game keep its frame rate and frame-time quality? | MangoHud CSV: mean FPS, 1 % low FPS, p99 frame time |
| Did the GPU behave differently? | `nvidia_smi_*.csv`: utilisation, clocks, power, temperature |
| What did the AI pay for the isolation? | `ai_requests.csv` and `ai_workload.log`: throughput and latency |
| Did the run start under comparable conditions? | `meta.json`: `started_hot`, `start_gpu_temp`, `start_cpu_temp` |

Suggested analysis, for each (model, intensity, device, game):

- **Check the partition first.** Drop or flag any run whose `game_affinity_*.txt` status is not `OK`, whose `ai_affinity.txt` shows another mask, or whose Ollama checks failed.
- Use the **raw frame times** from MangoHud, discard a warm-up at the start of the game window, and report the mean and a 95 % confidence interval across the 5 runs.
- **Game degradation** is the percentage drop in 1 % low FPS and the rise in p99 frame time relative to the unpinned co-run of Experiment 1 with the same combination.
- **AI cost** is the change in throughput and latency relative to the same Experiment 1 combination.
- If degradation is gone with a clean split, CPU scheduling contention was the cause. If some remains, the cause is in resources that core pinning does not separate (see below).

Points to keep in mind when interpreting results:

- **There is no pinned game-alone control in this sweep.** Confining a game to two or three logical CPUs can cost performance by itself, so an improvement or a loss against Experiment 1 mixes isolation with the effect of the small partition. The `idle` intensity (model loaded, no requests) is the nearest control inside the sweep. A pinned game-alone run can be made by calling `run_game.py <game> 240 --game-cores <mask>` directly.
- **The partitions are small.** Most games get one physical core (two SMT threads), Xonotic one and a half. Compare with the demand measured in Experiment 2.
- **Isolation is only at the CPU-core level.** The L3 cache, memory bandwidth, the package's power and thermal budget, and (for `gpu` placements) the GPU are still shared.
- **Not everything is pinned.** The controller, the desktop compositor, interrupts and kernel threads can run on any core. `cpu_log.py` and the `nvidia-smi` sidecar are free too, unless `--telemetry-cpu` is used. The partition check compares the telemetry CPU with the **game's** CPUs only, so the telemetry CPU normally lands inside the AI partition. Pick it deliberately and use the same one in every condition.
- **Intensity is a sleep, not a request rate.** The achieved rate depends on latency, so use `ai_requests.csv` for the real throughput.
- **A cooler start can raise boost clocks.** Check `start_gpu_temp` and `start_cpu_temp` before crediting a difference to pinning. Experiment 1 started games against a warm, steady-state AI; here the machine is cooled first.
- **Frame caps limit what degradation can show** for the capped games (Luanti 60 fps, SuperTuxKart 144 fps with vsync).
- **Qwen is only as pinned as the server you started.** The scripts verify it, but a server restarted between batches is the operator's responsibility.

---

## 8. Practical notes for a third party repeating the run

1. Put `run_task.py`, `run_task_ollama.py`, `run_game.py`, `cpu_partition.py` and `cpu_log.py` in one folder, separate from the Experiment 1 and 2 folders, and run from there.
2. Edit the Xonotic and Luanti paths in `run_game.py` if your install locations differ.
3. **Check your CPU layout before reusing the masks.** Run `./cpu_partition.py 0-2 4-15` (and `0-1 2-15`); it should print `OK`. On a machine with a different SMT layout the preflight will refuse to start, which is the intended behaviour.
4. Look at the plan first with `python run_task.py --dry-run`.
5. Do a short test, for example `python run_task.py --runs 1 --games glmark2 --models cnn --intensities high --devices gpu --duration 60`, and confirm that `run_1/` contains a MangoHud CSV, an `nvidia_smi` CSV, `game_affinity_*.txt` with status `OK`, `ai_affinity.txt`, `ai_workload.log`, `ai_requests.csv`, `cpu_percore.csv` and `meta.json`.
6. Verify the pinning on a real run: in `cpu_percore.csv`, the game's load should sit on CPUs `0-1` (or `0-2`) and the AI's load on the AI CPUs.
7. For qwen, run one batch per AI core set, restarting `ollama serve` with the matching `taskset` between batches, and use a separate pass (with the GPU hidden) for the CPU placement.
8. The cnn and tinybert sweep takes more than a day and a half. Use `--runs`, `--models`, `--intensities`, `--devices` and `--games` to cut it down, and run the most informative combinations first.
9. Keep the machine in the same state across runs (AC power, power mode, nothing else running). The setup used for the reported runs is in Appendix A.

---

# Appendix A — Test machine and configuration (Experiments 2 and 3)

Everything below was recorded for the Experiment 2 runs, and Experiment 3 was run on the same machine and configuration.

### A.1 System

| Item | Value |
|------|-------|
| Laptop CPU | Intel Core i5-13450HX (6 P-cores with 2 threads each, 4 E-cores) |
| Laptop GPU | NVIDIA RTX 4050 (plus the Intel integrated GPU, which is not used for the games) |
| Kernel | `7.0.0-38-generic` (`uname -r`) |
| NVIDIA driver | 595.99.02 |
| CUDA version (from `nvidia-smi`) | 13.2 |
| Power source | Plugged into AC for every run |
| Laptop power mode | Performance (LOQ performance mode) |
| Resting temperature before each run | 50 °C |

### A.2 Commands run before the experiment

These set the CPU to a fixed performance mode and locked the GPU graphics clock:

```bash
sudo cpupower frequency-set -g performance      # CPU governor: performance
sudo nvidia-smi -lgc 2700,2700                  # lock GPU graphics clock to 2700 MHz (min,max)
```

### A.3 CPU layout

16 logical CPUs on 10 physical cores: six P-cores with two threads each (SMT) and four single-thread E-cores.

| Type | Logical CPUs | Physical cores | Threads per core | Max MHz | Min MHz |
|------|--------------|----------------|------------------|---------|---------|
| P-core | (0,1), (2,3), (4,5), (6,7), (8,9), (10,11) | 6 | 2 | 4600 | 800 |
| E-core | 12, 13, 14, 15 | 4 | 1 | 3400 | 800 |

This matches the defaults in `demand_profile.py` (`--p-cpus 0-11`, `--e-cpus 12-15`). It was checked with:

```bash
lscpu -e=CPU,CORE,SOCKET,MAXMHZ,MINMHZ
for c in 0 2 4 6 8 10 12 13 14 15; do echo -n "cpu$c siblings: "; cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list; done
```

Reported thread siblings: `0-1`, `2-3`, `4-5`, `6-7`, `8-9`, `10-11` for the P-cores, and a single CPU each for 12, 13, 14 and 15.

### A.4 Games

| Game | Version | Resolution | Vsync | Frame cap |
|------|---------|------------|-------|-----------|
| Xonotic | git build `bd6e300` (Wed Aug 5 01:17:12 2026 +1000) | 1920×1080 | off | uncapped |
| glmark2 | 2023.01 | 1920×1080 | not present | uncapped |
| SuperTuxKart | 1.5 | 1920×1080 | on | 144 fps |
| Luanti | 5.18.0-dev-b4eb8b9-dirty (Linux) | window set by the game (launched with `--go --random-input`) | off | 60 fps |

Luanti build details: LuaJIT 2.1.1761786044 (OpenResty), built by GCC 15.2, running on Linux/7.0.0 x86_64, `BUILD_TYPE=Release`, `RUN_IN_PLACE=1`, `USE_CURL=1`, `USE_GETTEXT=1`, `USE_SOUND=1`, `STATIC_SHAREDIR="."`, `STATIC_LOCALEDIR="locale"`.

