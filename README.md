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
| **glmark2** | Fullscreen-style window 1920×1080 | System `glmark2` binary |
| **Luanti** | `--go --random-input` (starts a world and injects random input) | Local install path is hardcoded |
| **SuperTuxKart** | Demo mode: 8 karts, 100 laps, 1920×1080 | System `supertuxkart` binary |

The game is **always** forced onto the discrete NVIDIA GPU via PRIME render offload:

- `__NV_PRIME_RENDER_OFFLOAD=1`
- `__GLX_VENDOR_LIBRARY_NAME=nvidia`

So in both placement modes below, the **game stays on the GPU**. Only the AI job moves.

### 1.2 Three AI workloads

| Name | Model and data | How a “request” is defined |
|------|----------------|----------------------------|
| **cnn** | CIFAR-10 ResNet20 (`chenyaofo/pytorch-cifar-models`, pretrained). Up to 100 batches of CIFAR-10 test images (batch size 1024) are preloaded. | One forward pass of a randomly chosen batch, with `torch.cuda.synchronize()` when on GPU so latency is real wall time |
| **tinybert** | `prajjwal1/bert-tiny` on GLUE SST-2 (first 1000 validation sentences) | One forward pass of up to 32 randomly sampled sentences, tokenised to max length 128 |
| **qwen** | `qwen:0.5b` served by a local **Ollama** process (`http://localhost:11434`). Prompts are random instructions from Databricks Dolly-15k | One HTTP generate call (non-streaming). Timeout 120 s. Token count comes from Ollama’s `eval_count` |

CNN and TinyBERT are ordinary PyTorch processes in `run_ai_workload.py`. Qwen is different: this script only sends HTTP requests; the actual inference runs inside Ollama’s own server, which the `--device` flag **does not** move.

### 1.3 Two placement modes

These are the two “modes” of the experiment. They apply cleanly to **cnn** and **tinybert**.

| Mode | Folder / combo name suffix | What the AI does | What the game does |
|------|----------------------------|------------------|--------------------|
| **Shared GPU** | `_gpu` | PyTorch uses CUDA (`--device gpu`) | NVIDIA GPU via PRIME |
| **AI on CPU** | `_cpu` | PyTorch uses CPU (`--device cpu`) | NVIDIA GPU via PRIME (unchanged) |

The first mode is contention on the GPU (and whatever CPU the game and Python still use). The second mode is contention mainly on the CPU, while the GPU is left to the game.

**Qwen:** `run_batch.py` does **not** create `_gpu` / `_cpu` variants. Combination folders are named `qwen_<intensity>_ollama`. Placement is whatever the **already running** Ollama server was started with; the `--device` flag on `run_ai_workload.py` does not move it.

To get the same two modes as CNN/TinyBERT, **offload or bind the Ollama server yourself, then start `run_batch.py`**. Stop any existing `ollama` process first so the new launch is the one that serves `localhost:11434`.

Ollama on the GPU (shared with the game):

```
<cmd>
```

Ollama offloaded off the GPU (AI on CPU, game keeps the GPU):

```
<cmd>
```

Leave that server running for the whole Qwen set. Switching mid-sweep would mix GPU and CPU traces in the same `_ollama` folders.

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

Default full sweep size:

- cnn: 4 intensities × 2 modes = 8 combinations  
- tinybert: 8 combinations  
- qwen: 4 intensities × 1 (`ollama`) = 4 combinations  
- **20 combinations** × 4 games × 5 runs = **400 game runs**

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

These are the things the operator (or the lab machine) must have in place. They are not generated by the experiment.

### 3.1 Machine and software

- Linux (per-core logging reads `/proc/stat`)
- NVIDIA driver so `nvidia-smi` can read temperature, utilisation, power, clocks, and VRAM
- **MangoHud** on `PATH` (wraps the game and writes FPS / frame-time CSVs)
- `timeout` on `PATH` (preflight check; the current `run_game.py` does not wrap the game with `timeout`, but `run_batch.py` still requires the binary)
- Python with `torch`, `torchvision`, `transformers`, `datasets`, `requests`
- CUDA visible to that Python if any `_gpu` combinations are requested (otherwise those combinations would silently fall back to CPU; preflight refuses to start in that case)
- For Qwen: Ollama listening on `http://localhost:11434` with model `qwen:0.5b` pulled. Start that server in GPU or CPU placement **before** the sweep (commands in §1.3; fill in `<cmd>`)

### 3.2 Game binaries

Paths are in `GAMES_CONFIG` inside `run_game.py`. Xonotic and Luanti currently point at a local disk (`/run/media/tanish/...`). Those paths must be edited for another machine. glmark2 and SuperTuxKart are expected as system commands.

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
| `--cooldown-timeout` | 600 | Max wait for either cooldown |
| `--cpu-interval` | 30 | Per-core CPU sample period in milliseconds |
| `--results-dir` | `results` | Where combination folders go |
| `--per-core-dir` | `per_core_result` | Where per-core CSVs go |
| `--ai-script` / `--ai-python` | local script / current interpreter | If the AI environment lives elsewhere |
| `--fresh` | off | Ignore `state.json` and redo everything |
| `--dry-run` | off | Print the plan only |

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
   - … repeat until 5 iterations (no cooldown after the last run of the last iteration).
7. Each `run_game.py` call:
   - optional padding is **0** when called from the batch (no extra 10 s idle),
   - starts MangoHud around the game,
   - starts an `nvidia-smi` sidecar logger,
   - kills the game after `--duration` seconds,
   - waits 3 s for MangoHud to flush, then stops `nvidia-smi`.
8. Stop the per-core logger (SIGINT so the CSV is flushed).
9. Move every **new** file that appeared under `mangohud_raw/<game>/` into `results/<combo>/<game>/`.
10. Stop the AI (SIGINT so it prints throughput/latency summary into the log).
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
  ai_requests_<model>_<intensity>_<device>.csv   (next to run_ai_workload.py, see below)
```

`<device>` is `gpu`, `cpu`, or `ollama`.


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

Captured stdout/stderr of `run_ai_workload.py` for the whole time the AI was alive (warmup + baseline + all 20 game runs). Contains load messages, per-request errors (first five), and on SIGINT the **summary**: total / fulfilled / failed requests, throughput, mean latency, min / p50 / p95 / p99 / max latency.

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
| `timestamp` | Unix epoch seconds (same clock family as AI `epoch_s`) |
| `elapsed_s` | Seconds since this logger started |
| `cpu0`, `cpu1`, … | Busy percentage of that logical core over the last interval (100 × busy_ticks / total_ticks; idle includes iowait) |

If a file already exists from a previous attempt, it is renamed `*_prev_YYYYMMDD_HHMMSS.csv` instead of overwritten.

`run_batch.py` **requires** `cpu_log.py` in the **same folder** (`exp1/cpu_log.py`).

### 5.8 `ai_requests_<model>_<intensity>_<device>.csv`

Per-request AI log from `run_ai_workload.py`. Default path is the **current working directory of the AI process**. `run_batch.py` starts that process with `cwd` = the directory of `run_ai_workload.py`, and it does **not** pass `--log-csv`, so this file is **not** moved into `results/<combo>/`. Look next to the AI script (or in `./data`’s parent). It is overwritten on the next combination that uses the same model/intensity/device name.

| Column | Content |
|--------|---------|
| `request_id` | 1, 2, 3, … |
| `epoch_s` | Unix time when the request started (align with MangoHud / nvidia-smi / per-core `timestamp`) |
| `elapsed_s` | Seconds since the AI loop started |
| `latency_ms` | Wall time of that request (GPU path includes `cuda.synchronize`) |
| `ok` | 1 if the request succeeded, 0 otherwise |
| `tokens` | For Qwen, Ollama `eval_count`; empty for cnn/tinybert |

Idle intensity writes almost nothing useful here (the loop does not issue requests).

On shutdown the same numbers are summarised in `ai_workload.log`.

### 5.9 `mangohud_raw/`

Not a result folder. It is a staging area. After each combination, newly created files are moved out. Leftovers mean a crash happened before collect, or a manual `run_game.py` was used.

### 5.10 CIFAR-10 `data/` next to the AI script

Created on first CNN run (`torchvision` download). Dataset cache, not a benchmark metric.

---

## 6. Scripts in `exp1/` (every file)

### `run_batch.py` — sweep controller

This is the script a third party should run for the published design. It does not render frames or run neural nets itself. It:

- builds the list of combinations (Qwen forced to device `ollama`)
- checks games, MangoHud, `nvidia-smi`, CUDA, Ollama, and `cpu_log.py` (preflight)
- starts/stops AI and CPU logger
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

Holds `GAMES_CONFIG` (the four games and their command lines). Sets PRIME offload, builds `MANGOHUD_CONFIG`, launches `mangohud` + game in a new process session, starts `nvidia-smi -lms 100` into `mangohud_raw/<game>/nvidia_smi_<timestamp>.csv`, waits `duration` seconds, SIGINT then SIGKILL the game group, sleeps 3 s, stops nvidia-smi.

`--padding` default 10 s before and after is only for manual use. The batch always passes `--padding 0` so the only waits between games are the temperature cooldowns.

`--pci-dev` can pin MangoHud to a PCI id if the HUD is logging the wrong GPU.

### `run_ai_workload.py` — background AI job

Loads one of cnn / tinybert / qwen, then issues requests until `duration` or SIGINT. Intensity is the sleep table in §1.4. `--device` selects PyTorch CPU vs CUDA; for Qwen it only prints a warning that Ollama is independent.

Writes `ai_requests_*.csv` and a latency summary to stdout (captured as `ai_workload.log`).

### `cpu_log.py` — per-core utilisation

Lives next to `run_batch.py`. Reads `/proc/stat` and writes `timestamp,elapsed_s,cpu0,...`. SIGINT flushes and exits. Sampling interval comes from `--cpu-interval` (30 ms).

### `results_gpu_cpu_laptop.zip` / `plots_gpu_cpu_laptop.zip`

Packed outputs from a laptop run of this experiment (raw results and derived plots). They are not used as inputs by the scripts.

---

## 7. How to read the two modes in the data

For a fixed game, intensity, and AI model (cnn or tinybert):

| Compare | Folder A | Folder B | Interpretation |
|---------|----------|----------|----------------|
| Shared GPU vs AI-on-CPU | `*_gpu` | `*_cpu` | Effect of putting inference on the same GPU as the game versus leaving the GPU to the game |
| Intensity | `*_high_*` vs `*_mid_*` vs `*_low_*` vs `*_idle_*` | same | Effect of how often inference runs; `idle` is “model resident, no traffic” |
| Game | subfolders `xonotic`, `glmark2`, … | same combo | Different rendering/CPU behaviour under the same AI |

Use MangoHud for FPS/frame time, `nvidia_smi_*.csv` for discrete-GPU truth, `per_core_result/*.csv` for CPU-core busy %, and `ai_requests_*.csv` / `ai_workload.log` for inference latency and throughput. Align clocks with `epoch_s` / `timestamp`.

---

## 8. Practical notes for a third party repeating the run

1. Work in `exp1/` so relative output paths (`mangohud_raw`, `results`) stay together.
2. Edit game paths in `run_game.py` before a long sweep.
3. For Qwen, start Ollama in the intended placement (commands in §1.3) **before** `run_batch.py`.
4. Do a 30 s single game (`python run_game.py glmark2 30`) and confirm both a MangoHud CSV and an `nvidia_smi_*.csv` appear.
5. A full default sweep is many hours: 20 combinations × (300 s warmup + 240 s baseline + 20 × 240 s games + cooldowns). Cooldown time is not included in the printed estimate.
6. Do not `pip install -r` the repository-root `requirements.txt`; it is a full OS package dump, not an experiment environment file.
