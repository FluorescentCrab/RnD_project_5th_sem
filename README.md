Experiment 1 — Unpinned co-run of games and AI workloads
Experiment 1 asks a single question:
> When a video game is already using the NVIDIA GPU, how do frame rate, GPU utilisation, power, temperature, and CPU use change if an AI inference job is running at the same time — either **on that same GPU**, or **entirely on the CPU**?
There is no CPU-core pinning in this experiment. The operating system scheduler places threads wherever it wants. Later experiments (not covered here) add pinning and game-only CPU demand measurements.
---
1. What is being tested
1.1 Four games
Every combination runs the same four titles. Each is launched in a repeatable, non-interactive mode so a human does not have to play.
Game	How it is driven	Notes
Xonotic	Plays a fixed demo (`demos/rnd_demo.dem`) at 1920×1080, uncapped FPS (`+cl_maxfps 0`)	Local install path is hardcoded in `run_game.py`
glmark2	Window at 1920×1080	System `glmark2` binary
Luanti	`--go --random-input` (starts a world and injects random input)	Local install path is hardcoded in `run_game.py`
SuperTuxKart	Demo mode: 8 karts, 100 laps, 1920×1080	System `supertuxkart` binary
The game is always forced onto the discrete NVIDIA GPU via PRIME render offload:
`__NV_PRIME_RENDER_OFFLOAD=1`
`__GLX_VENDOR_LIBRARY_NAME=nvidia`
So in both placement modes below, the game stays on the GPU. Only the AI job moves.
1.2 Three AI workloads
Name	Model and data	How a “request” is defined
cnn	CIFAR-10 ResNet20 (`chenyaofo/pytorch-cifar-models`, pretrained). The CIFAR-10 test set (10,000 images) is preloaded as 10 batches (nine of 1024 images and one of 784).	One forward pass of a randomly chosen batch, with `torch.cuda.synchronize()` when on GPU so latency is real wall time
tinybert	`prajjwal1/bert-tiny` on GLUE SST-2 (first 1000 validation sentences)	One forward pass of up to 32 randomly sampled sentences, tokenised to max length 128
qwen	`qwen:0.5b` served by a local Ollama process (`http://localhost:11434`). Prompts are random instructions from Databricks Dolly-15k	One HTTP generate call (non-streaming). Timeout 120 s. Token count comes from Ollama’s `eval_count`
CNN and TinyBERT are ordinary PyTorch processes started by `run_ai_workload.py`. Qwen is different: that script only sends HTTP requests; the actual inference runs inside Ollama’s own server.
1.3 Two placement modes
Mode	Folder / combo name suffix	What the AI does	What the game does
Shared GPU	`_gpu`	AI runs on the NVIDIA GPU	NVIDIA GPU via PRIME
AI on CPU	`_cpu`	AI runs on the CPU	NVIDIA GPU via PRIME (unchanged)
The first mode is contention on the GPU (and whatever CPU the game and Python still use). The second mode is contention mainly on the CPU, while the GPU is left to the game.
CNN and TinyBERT: `run_batch.py` creates both modes automatically by passing `--device gpu` or `--device cpu` to `run_ai_workload.py`.
Qwen (Ollama): the two modes are not created by `run_batch.py`. It only creates folders named `qwen_<intensity>_ollama`, and the `--device` flag does not move Ollama. Placement is whatever the already running Ollama server was started with. The procedure is therefore done in two passes:
Pass 1 — Ollama on the GPU (shared with the game). Start Ollama normally and leave it running for the whole Qwen set:
```
   ollama serve
   ```
Run the Qwen sweep and let it finish (see §3.3).
Pass 2 — Ollama off the GPU (AI on CPU). Stop the running Ollama server, restart it so that inference runs on the CPU, and run the Qwen sweep again:
```
   <cmd>
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
Note on Qwen logs: `run_batch.py` always passes `--device gpu` to `run_ai_workload.py` for Qwen, so `ai_workload.log` shows “Device: cuda” for Qwen. This label only reflects PyTorch’s device choice, which Qwen does not use, and says nothing about where Ollama is running. For the same reason, the “Ollama uses its own background server” warning in `run_ai_workload.py` (printed only when `--device cpu` is passed) never appears in Qwen logs. Record the Ollama placement yourself (the folder name above does this).
1.4 Request intensity (how hard the AI is driven)
Between successive AI requests the script sleeps:
Intensity	Sleep after each request
`high`	0.05 s
`mid`	0.5 s
`low`	2.0 s
`idle`	no requests; the process just sleeps 1 s at a time (model may still sit in memory / on the GPU)
The default sweep uses all four intensities.
1.5 Repeats
For each combination of (AI model × intensity × placement mode):
The AI job is started once and left running.
Each of the four games is run 5 times.
Each game run lasts 240 seconds (4 minutes) by default.
Default sweep size (one run of `python run_batch.py`):
cnn: 4 intensities × 2 modes = 8 combinations
tinybert: 8 combinations
qwen: 4 intensities × 1 (`ollama`) = 4 combinations
20 combinations × 4 games × 5 runs = 400 game runs
Running the Qwen sweep a second time for Ollama on CPU (§1.3) adds 4 more combinations and 80 more game runs.
Expected duration. Each combination takes about 300 s warmup + 240 s baseline + 20 × (240 s + ~10 s for MangoHud flush and shutdown) ≈ 5,540 s, or roughly 1 h 32 m. The full 20-combination sweep is therefore roughly 31 hours, and cooldown waits come on top of that (the time estimate printed by `run_batch.py` does not include them).
---
2. Temperature control
Temperature is used so that one hot run does not bias the next. There are two different targets.
2.1 Cooldown between game runs (inside one combination)
After the AI is loaded, the controller samples GPU temperature with `nvidia-smi` for 240 s (default `--baseline-duration`) while the AI is already running and no game is running. The mean of those samples is the baseline.
The target passed into `run_game_batch.py` is:
```text
cooldown_target = ceil(baseline_avg + temp_margin)
```
`--temp-margin` default is 2 °C
The value is rounded up to a whole degree because `run_game_batch.py` only accepts an integer `--target-temp`
Between every game run except the last in that combination, the batch script waits until:
```text
GPU temperature ≤ cooldown_target
```
Polling is every 5 s. If the GPU has not reached the target within 600 s (`--cooldown-timeout`), the next run starts anyway.
This target is not a fixed 45 °C. It tracks how hot the AI alone makes the GPU, plus 2 °C, so cooldown returns the machine to “AI-only steady state”, not to a cold idle.
The measured `baseline_avg` and `cooldown_target` are written into that combination’s `meta.json`.
2.2 Cooldown between combinations
After a combination finishes (AI stopped, logs collected), the controller waits until the GPU is ≤ 45 °C (`--idle-temp`, default 45) before starting the next model/intensity/mode. Same 5 s poll and 600 s timeout.
When `run_game_batch.py` is used by itself (not through `run_batch.py`), its own default `--target-temp` is 45 °C.
---
3. Inputs you must provide
3.1 Machine and software
Linux (per-core logging reads `/proc/stat`)
NVIDIA driver so `nvidia-smi` can read temperature, utilisation, power, clocks, and VRAM
MangoHud on `PATH` (wraps the game and writes FPS / frame-time CSVs)
`timeout` on `PATH`. The preflight check in `run_batch.py` requires the binary to exist, although `run_game.py` itself does not wrap the game with it
Python with `torch`, `torchvision`, `transformers`, `datasets`, `requests`
CUDA visible to that Python if any `_gpu` combinations are requested (otherwise those combinations would silently fall back to CPU; preflight refuses to start in that case)
For Qwen: Ollama listening on `http://localhost:11434` with model `qwen:0.5b` pulled, started in the intended placement before each Qwen pass (see §1.3)
3.2 Game binaries
Paths are in `GAMES_CONFIG` inside `run_game.py`. Xonotic and Luanti currently point at a local disk (`/run/media/tanish/...`). On another machine these paths would have to be changed in `run_game.py`. glmark2 and SuperTuxKart are expected as system commands.
Launch arguments (resolution, demo file, random input, demo laps) are part of the experimental input: they define the visual load. Do not change them if you want comparable traces.
3.3 Command-line inputs to the controller
Typical full run, from the `exp1/` directory (working directory must contain `run_game.py`):
```bash
python run_batch.py
```
Useful knobs (all optional; defaults match the design above):
Argument	Default	Meaning
`--runs`	5	Repeats per game per combination
`--duration`	240	Seconds each game is allowed to run
`--models`	cnn tinybert qwen	Which AI jobs
`--intensities`	high mid low idle	Request rates
`--devices`	gpu cpu	Placement modes for cnn/tinybert
`--warmup`	300	Seconds to wait after starting the AI before treating it as loaded
`--ready-timeout`	120	Extra wait until the log contains `Starting <model>`
`--baseline-duration`	240	Seconds of GPU-temp sampling for the per-combination baseline
`--temp-margin`	2	°C added to that baseline (then rounded up)
`--idle-temp`	45	GPU °C to reach before the next combination
`--poll-interval`	5	Seconds between GPU temperature checks
`--cooldown-timeout`	600	Max wait for either cooldown
`--cpu-interval`	30	Per-core CPU sample period in milliseconds
`--results-dir`	`results`	Where combination folders go
`--per-core-dir`	`per_core_result`	Where per-core CSVs go
`--ai-script` / `--ai-python`	local script / current interpreter	If the AI environment lives elsewhere
`--fresh`	off	Ignore `state.json` and redo everything
`--dry-run`	off	Print the plan only
`--skip-preflight`	off	Skip the startup sanity checks
Standalone (not the full sweep):
```bash
python run_game.py xonotic 240
python run_game_batch.py 5 240 --target-temp 45
python run_ai_workload.py cnn 240 high --device gpu
```
---
4. What a combination does, in order
For one name such as `cnn_high_gpu`:
Create `results/cnn_high_gpu/` (if that folder already exists from a crashed attempt, it is renamed with a `_prev_<timestamp>` suffix so old files are not mixed with new ones).
Start `run_ai_workload.py` in the background for a very long duration (24 h). The batch script will stop it with SIGINT. Stdout/stderr go to `ai_workload.log`.
Wait `--warmup` seconds, then until the log shows the model has started serving.
Sample GPU temperature for `--baseline-duration` seconds; compute `cooldown_target`.
Start `cpu_log.py` (must sit next to `run_batch.py`). It writes `per_core_result/cnn_high_gpu.csv`.
Run `run_game_batch.py 5 240 --target-temp <cooldown_target>`:
iteration 1: xonotic → cooldown → glmark2 → cooldown → luanti → cooldown → supertuxkart → cooldown
… repeat until 5 iterations (no cooldown after the very last run).
Each `run_game.py` call:
uses padding 0 when called from the batch (no extra 10 s idle),
starts MangoHud around the game,
starts an `nvidia-smi` sidecar logger,
kills the game after `--duration` seconds,
waits 3 s for MangoHud to flush, then stops `nvidia-smi`.
Stop the per-core logger (SIGINT so the CSV is flushed).
Move every new file that appeared under `mangohud_raw/<game>/` into `results/<combo>/<game>/`.
Stop the AI (SIGINT so it prints its throughput/latency summary into the log).
Write `meta.json`. Mark the combination completed in `results/state.json`.
Cool the GPU to `--idle-temp` (45 °C) before the next combination.
If you interrupt with Ctrl+C, re-run the same command. Completed names in `state.json` are skipped. Failed names can be retried. Partial MangoHud files from the interrupted combination are still collected.
---
5. Outputs — everything that is saved
5.1 Directory layout after a sweep
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
5.2 `results/state.json`
Progress file for the whole sweep. Fields:
`completed` — list of combination names that finished
`failed` — map of combination name → error string (AI died, never became ready, `run_game_batch.py` non-zero exit, GPU temp unreadable, …)
This is the resume mechanism. It is not per-frame data.
5.3 `results/<combo>/meta.json`
One JSON object describing that combination. Typical keys:
Key	Meaning
`combo`	Name, e.g. `tinybert_mid_cpu`
`model`, `intensity`, `device`	The three factors
`runs_per_game`	Repeat count (5)
`duration_s`	Game length (240)
`started`, `finished`	Local ISO timestamps
`baseline_avg`	Mean GPU °C during the AI-only baseline window
`cooldown_target`	Integer °C used between game runs
`per_core_csv`	Path of the per-core log for this combo
`cpu_interval_ms`	Sample interval (30)
`csv_counts`	Per game, how many non-empty MangoHud CSVs were collected (excludes `*_summary.csv`)
If `csv_counts` for a game is less than 5, that game produced fewer HUD logs than repeats (crash, HUD failed to flush, empty file).
5.4 `results/<combo>/ai_workload.log`
Captured stdout/stderr of `run_ai_workload.py` for the whole time the AI was alive (warmup + baseline + all 20 game runs). Contains load messages, per-request errors (first five), and on SIGINT the summary: total / fulfilled / failed requests, throughput, mean latency, min / p50 / p95 / p99 / max latency. This is the record of AI-side performance for the combination.
5.5 MangoHud CSV (`results/<combo>/<game>/*.csv`)
Primary game trace. MangoHud is configured with:
`autostart_log=1`
`log_interval=100` (100 ms)
`log_duration = duration - 10` (230 s of a 240 s run), so the file is closed before the process is killed
`output_folder = mangohud_raw/<game>` (later moved)
sensors: `cpu_stats`, `gpu_stats`, `cpu_temp`, `gpu_temp`, `cpu_power`, `gpu_power`, `gpu_core_clock`, `gpu_mem_clock`, `vram`, `ram`, `core_stats`
`gpu_list=0` unless disabled
Expect one main CSV per successful run, plus often a `*_summary.csv` (MangoHud’s own averages). The collector counts only non-empty files that do not end in `_summary.csv` when checking “5 files per game”.
Typical columns (names follow MangoHud’s version): FPS, frame time, CPU load, GPU load, CPU/GPU temperature, CPU/GPU power, core and memory clocks, VRAM, RAM. Use these for frame-time and GPU-side plots.
Caveat: on hybrid laptops MangoHud can attach to the iGPU. That is why a second logger exists.
5.6 `nvidia_smi_YYYYMMDD_HHMMSS.csv` (same game folders)
Sidecar from:
```text
nvidia-smi --query-gpu=timestamp,utilization.gpu,temperature.gpu,power.draw,clocks.gr,clocks.mem,memory.used
           --format=csv -lms 100
```
One row about every 100 ms, from NVIDIA’s NVML, independent of MangoHud. Columns:
Column	Content
timestamp	Wall-clock time from `nvidia-smi`
utilization.gpu	GPU compute utilisation (%)
temperature.gpu	GPU °C
power.draw	Board power
clocks.gr	Graphics clock
clocks.mem	Memory clock
memory.used	VRAM used
One file per game run. Align with MangoHud using timestamps.
5.7 `per_core_result/<combo>.csv`
Written by `cpu_log.py` for the whole game-batch window of that combination (all four games × 5 runs, including cooldowns between them). Sampled from `/proc/stat` every 30 ms by default.
Column	Content
`timestamp`	Unix epoch seconds
`elapsed_s`	Seconds since this logger started
`cpu0`, `cpu1`, …	Busy percentage of that logical core over the last interval (100 × busy_ticks / total_ticks; idle includes iowait)
If a file already exists from a previous attempt, it is renamed `*_prev_YYYYMMDD_HHMMSS.csv` instead of overwritten.
`run_batch.py` requires `cpu_log.py` in the same folder as itself.
5.8 `mangohud_raw/`
Not a result folder. It is a staging area. After each combination, newly created files are moved out. Leftovers mean a crash happened before collection, or a manual `run_game.py` was used.
5.9 CIFAR-10 `data/` next to the AI script
Created on first CNN run (`torchvision` download). Dataset cache, not a benchmark metric.
---
6. Scripts in `exp1/` (every file)
`run_batch.py` — sweep controller
This is the script a third party should run for the published design. It does not render frames or run neural nets itself. It:
builds the list of combinations (Qwen forced to device `ollama`)
checks games, MangoHud, `nvidia-smi`, `timeout`, CUDA, Ollama, and `cpu_log.py` (preflight)
starts/stops the AI job and the CPU logger
measures baseline GPU temperature and computes `cooldown_target`
calls `run_game_batch.py`
moves logs into `results/`
writes `meta.json` and `state.json`
cools to 45 °C between combinations
`run_game_batch.py` — five repeats of all four games
Arguments: `n` (repeats), `t` (seconds), `--target-temp`, `--poll-interval`, `--cooldown-timeout`.
Nested loops: for `i` in 1..n, for each game in `GAMES_CONFIG` order (xonotic, glmark2, luanti, supertuxkart), call `run_game.py <game> t --padding 0`, then wait for GPU ≤ `--target-temp` unless this was the last run.
When called from `run_batch.py`, `--target-temp` is the dynamic AI-baseline + 2 °C value, not 45.
`run_game.py` — one game, one timed run
Holds `GAMES_CONFIG` (the four games and their command lines). Sets PRIME offload, builds `MANGOHUD_CONFIG`, launches `mangohud` + game in a new process session, starts `nvidia-smi -lms 100` into `mangohud_raw/<game>/nvidia_smi_<timestamp>.csv`, waits `duration` seconds, sends SIGINT then SIGKILL to the game's process group, sleeps 3 s, stops `nvidia-smi`.
`--padding` default 10 s before and after is only for manual use. The batch always passes `--padding 0` so the only waits between games are the temperature cooldowns.
`--pci-dev` can pin MangoHud to a PCI id if the HUD is logging the wrong GPU.
`run_ai_workload.py` — background AI job
Loads one of cnn / tinybert / qwen, then issues requests until `duration` or SIGINT. Intensity is the sleep table in §1.4. `--device` selects PyTorch CPU vs CUDA for cnn and tinybert; it has no effect on where Qwen runs (see §1.3). On exit it prints a latency and throughput summary to stdout, which `run_batch.py` captures as `ai_workload.log`.
`cpu_log.py` — per-core utilisation
Lives next to `run_batch.py`. Reads `/proc/stat` and writes `timestamp,elapsed_s,cpu0,...`. SIGINT flushes and exits. Sampling interval comes from `--cpu-interval` (30 ms).
---
7. How to read the modes in the data
For a fixed game, intensity, and AI model:
Compare	Folder A	Folder B	Interpretation
Shared GPU vs AI-on-CPU (cnn, tinybert)	`*_gpu`	`*_cpu`	Effect of putting inference on the same GPU as the game versus leaving the GPU to the game
Shared GPU vs AI-on-CPU (qwen)	Ollama-on-GPU pass folders	Ollama-on-CPU pass folders	Same comparison, using the two separate Qwen passes from §1.3
Intensity	`*_high_*` vs `*_mid_*` vs `*_low_*` vs `*_idle_*`	same	Effect of how often inference runs; `idle` is “model resident, no traffic”
Game	subfolders `xonotic`, `glmark2`, …	same combo	Different rendering/CPU behaviour under the same AI
Use MangoHud for FPS/frame time, `nvidia_smi_*.csv` for discrete-GPU truth, `per_core_result/*.csv` for CPU-core busy %, and `ai_workload.log` for inference latency and throughput.
---
8. Practical notes for a third party repeating the run
Work in `exp1/` so relative output paths (`mangohud_raw`, `results`) stay together.
Change the Xonotic and Luanti paths in `run_game.py` if your install locations differ.
For Qwen, start Ollama in the intended placement (§1.3) before `run_batch.py`, and use separate results folders for the GPU and CPU passes.
Do a 30 s single game (`python run_game.py glmark2 30`) and confirm both a MangoHud CSV and an `nvidia_smi_*.csv` appear.
A full default sweep takes roughly 31 hours before cooldowns (§1.5).
Do not `pip install -r` the repository-root `requirements.txt`; it is a full OS package dump, not an experiment environment file.
---
---
Experiment 2 — CPU demand of each game (games run alone)
Experiment 2 asks a single question:
> How much CPU does each game need when it runs **by itself**, with no AI job alongside it?
No AI workload is started in this experiment. Each game is run alone, on the NVIDIA GPU, while the CPU is sampled at several time resolutions. The result is a per-game picture of how many cores' worth of CPU the game uses, how that splits between P-cores and E-cores, and how many threads and processes are behind it. It is the CPU-side baseline that the Experiment 1 co-runs can be read against.
As in Experiment 1, there is no CPU-core pinning: the scheduler places threads wherever it wants.
---
1. What is being tested
1.1 Four games
The same four titles as Experiment 1, launched in the same repeatable, non-interactive modes (resolution, demo file, random input and demo laps are the same). The frame cap and vsync state of each game during the runs are:
Game	Frame cap	Vsync	Notes
Xonotic	none (uncapped)	off	1920×1080, plays `demos/rnd_demo.dem`
glmark2	none (uncapped)	not present	1920×1080
Luanti	60 fps	off	`--go --random-input`
SuperTuxKart	144 fps	on	demo mode, 8 karts, 100 laps, 1920×1080
Frame caps matter when reading the results (see §7): a capped game's CPU demand is partly set by the cap, while an uncapped game takes as much CPU as it can use.
The game is forced onto the discrete NVIDIA GPU via PRIME render offload (`__NV_PRIME_RENDER_OFFLOAD=1`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`), so it does not render on the Intel integrated GPU.
1.2 What is measured
Four CPU measurements are taken during every run. They overlap on purpose, because each has a different resolution and a different view of the machine.
Measurement	Source	Resolution	What it sees
Demand profile (`demand.csv`)	`demand_profile.py` reading `/proc`	0.25 s	CPU time of every thread of every process in the game's session, as “cores-worth”, split into P-cores and E-cores
Per-thread log (`pidstat_raw.txt`)	`pidstat -t -u`	1 s	Per-thread CPU use of one game process, and which logical CPU each thread was on
Process tree (`process_tree.txt`)	`ps --forest` in a loop	10 s	Every process in the game's session with its cumulative CPU seconds
System per-core log (`cpu_log.csv`)	`cpu_log.py` reading `/proc/stat`	30 ms	Busy % of every logical CPU, for the whole machine
`demand_profile.py` and `pidstat` collect essentially the same kind of data (per-thread CPU time). They differ in sampling interval, and in that `demand_profile.py` covers the whole process session while `pidstat` watches a single process.
1.3 Repeats
For each game, the run is repeated 3 times (iterations), each lasting 240 seconds, with 10 seconds of padding before and after the game. That is 4 games × 3 iterations = 12 runs, executed game by game:
```text
xonotic iter 1,2,3 → glmark2 iter 1,2,3 → luanti iter 1,2,3 → supertuxkart iter 1,2,3
```
Each run takes roughly 263 s (10 s pre-padding + 240 s game + 3 s flush + 10 s post-padding) plus the cooldown wait, so the whole experiment is a little under an hour before cooldowns.
1.4 How this differs from Experiment 1
	Experiment 1	Experiment 2
AI job	Running during every game	None
Question	Effect of AI on game and GPU/CPU	CPU needed by each game alone
Controller	`run_batch.py`	`run_all.py`
Game launcher	`run_game.py`	`run_game_cpu_core.py`
Cooldown target	Adaptive (AI baseline + 2 °C)	Fixed 50 °C on CPU and GPU
Cooldown applied	Between runs	Before every run, including the first
Padding	0 (batch passes `--padding 0`)	10 s before and after
Data kept	MangoHud, `nvidia-smi`, per-core CSV, AI log	CPU data only (see §5)
---
2. Temperature control
Before every run, `run_all.py` waits until the CPU and the NVIDIA GPU are both at or below the target temperature:
Target: `--temp-target`, default 50 °C. The resting temperature used for the runs was 50 °C. A value of `0` disables the wait.
Polling every 5 s, with a status line about every 15 s.
Timeout: `--temp-timeout`, default 600 s. If the machine has not cooled in time, the run starts anyway and a warning is printed.
CPU temperature is read from `hwmon` (`coretemp`, `k10temp`, `zenpower` or `cpu_thermal`, preferring “Package id 0”, `Tctl` or `Tdie`), falling back to the `x86_pkg_temp` thermal zone. GPU temperature comes from `nvidia-smi`.
If one sensor cannot be found, a warning is printed and that sensor is ignored. If neither is found, no wait is done.
---
3. Inputs you must provide
3.1 Machine and software
Linux (reads `/proc` and `/sys`)
NVIDIA driver, with `nvidia-smi` available
MangoHud on `PATH` (the launcher still wraps the game with it)
`pidstat` on `PATH` (from the `sysstat` package)
`ps` and `bash` (used by the process-tree logger); `stdbuf` is used if present
Python 3 (standard library only for the four scripts in this experiment)
The four scripts must sit in the same folder: `run_all.py` imports `demand_profile.py`, and `run_game_cpu_core.py` looks for `demand_profile.py` next to itself. `run_all.py` also expects `cpu_log.py` and `run_game_cpu_core.py` there by default.
3.2 Game binaries and working directory
`run_game_cpu_core.py` has its own copy of `GAMES_CONFIG`; it does not import the one from `run_game.py`. Edit paths in this copy for Experiment 2.
Xonotic is given as a relative path (`../xonotic/xonotic/all`), so `run_all.py` must be started from a folder where that path resolves.
Luanti is still an absolute local path (`/run/media/tanish/...`). On another machine this path would have to be changed.
glmark2 and SuperTuxKart are expected as system commands.
As in Experiment 1, do not change launch arguments if you want comparable traces.
3.3 Command-line inputs
The settings used for the reported runs (240 s per run, 3 iterations, 10 s padding, 50 °C target) are the script defaults apart from the duration:
```bash
python run_all.py 240 -n 3
```
Arguments of `run_all.py`:
Argument	Default	Meaning
`duration` (positional)	—	Seconds per run, passed to `run_game_cpu_core.py`
`-n`, `--iterations`	3	Iterations per game
`-g`, `--games`	all four	Which games to run
`-o`, `--output`	`cpu_runs`	Output root folder
`--cpu-interval`	30	`cpu_log.py` interval in ms
`--padding`	10	Pre/post padding in seconds, passed to the game launcher
`--pidstat-interval`	1	`pidstat` interval in seconds
`--temp-target`	50	°C both CPU and GPU must reach before each run (0 disables)
`--temp-timeout`	600	Max seconds to wait for cooldown per run (0 = no limit)
`--cpu-script` / `--game-script`	scripts in the same folder	Paths to `cpu_log.py` / `run_game_cpu_core.py`
Standalone pieces:
```bash
python run_game_cpu_core.py glmark2 60                  # one game, outputs in mangohud_raw/glmark2/
python demand_profile.py --summary demand.csv           # p50/p95/max summary of an existing demand.csv
python cpu_log.py -i 30 -d 60 -o cpu_log.csv            # per-core logger by itself
```
`run_game_cpu_core.py` also accepts `--demand-dt` (default 0.25 s), `--pci-dev`, `--gpu-list` and `--debug`. `run_all.py` does not pass these on, so the sampler always runs at 0.25 s in a normal experiment.
---
4. What one run does, in order
For one game and iteration, such as `xonotic` iteration 1:
Cool down until CPU and GPU are ≤ `--temp-target` (§2).
Delete `mangohud_raw/<game>/` in the working directory, so no stale files remain from an earlier run.
Start `cpu_log.py` in the background, writing `<run_dir>/cpu_log.csv`.
Run `run_game_cpu_core.py <game> <duration> --padding 10 --pidstat-interval 1`, which does the following:
Wait the 10 s pre-run padding.
Launch `mangohud` + the game in a new process session (so the whole tree can be sampled and killed together).
Start `demand_profile.py` on that session (0.25 s sampling).
Start the process-tree logger (`ps --forest` every 10 s).
Start the `nvidia-smi` sidecar logger (100 ms).
Find the right process for `pidstat`. Some games start through a wrapper shell script (Xonotic's `all`) that sits idle while the real game runs as its child. A wrapper has one thread and a real game has several, so the script waits (up to 60 s) for a multi-threaded process and uses that one. Then start `pidstat -t -u -p <pid> 1`.
Wait `duration` seconds, then send SIGINT to the game's process group, and SIGKILL if it has not exited after 10 s.
Wait 3 s for MangoHud to flush, then stop `nvidia-smi`, `pidstat`, `demand_profile.py` and the process-tree logger.
Convert `pidstat_raw.txt` into `cpu_cores_used.csv` (distinct cores per timestamp) and print a short summary.
Wait the 10 s post-run padding.
Stop `cpu_log.py` with SIGINT so it flushes its CSV.
Keep three files from `mangohud_raw/<game>/` (`pidstat_raw.txt`, `demand.csv`, `process_tree.txt`) by copying them into the run folder, then delete everything else in `mangohud_raw/<game>/` (§5.3).
Print the demand summary and the process-tree check (§5.7).
Wait 2 s, then move on to the next iteration or game.
Because `cpu_log.py` is started before the game launcher and stopped after it, `cpu_log.csv` also contains the padding periods; `demand.csv` and `process_tree.txt` cover only the game's own lifetime.
---
5. Outputs — everything that is saved
5.1 Directory layout after a run
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
5.2 `demand.csv` — the main result
Written by `demand_profile.py` every 0.25 s. It samples `utime + stime` of every thread of every process in the game's session from `/proc`, so child processes and helper threads are included without needing any PIDs. Cores-worth means CPU seconds used per wall-clock second.
Column	Content
`epoch`	Wall-clock Unix seconds (same clock as `cpu_log.csv` `timestamp`)
`t`	Seconds since the sampler started
`total_cores`	Cores-worth used by the whole session
`p_cores`	Part of that on P-core logical CPUs (0–11)
`e_cores`	Part of that on E-core logical CPUs (12–15)
`busiest_thread_cores`	Cores-worth of the single busiest thread in that interval
`n_threads`	Threads seen
`n_procs`	Processes in the session
The process list is refreshed every 1 s to pick up child processes started later. The rate is computed from the real elapsed time of each interval, not the nominal 0.25 s. `python demand_profile.py --summary demand.csv` (and `run_all.py` after each run) prints the p50 / p95 / max of total cores-worth, and the p95 of P-cores, E-cores and the busiest thread, dropping the first 30 s of the run as warm-up (or nothing, if the run is shorter than 30 s).
5.3 What is deleted
Everything else the game launcher produces is deleted by `run_all.py`: the MangoHud CSVs, the `nvidia_smi.csv`, `pidstat_err.txt`, and the `cpu_cores_used.csv` that was just built from the pidstat log. So Experiment 2 has no FPS, GPU or power data, and the pidstat log is kept only in its raw form.
5.4 `pidstat_raw.txt`
Raw output of `pidstat -t -u -p <pid> 1` (per-thread CPU use of one process, 1 s interval, 24 h ISO timestamps, `LC_ALL=C`). Each row gives a thread's CPU use and the logical CPU it was running on at that moment. It is converted to a CSV afterwards in a separate Jupyter step (not part of this folder). That conversion follows the same idea as the `parse_pidstat` function in `run_game_cpu_core.py`: for each timestamp, the sorted list of distinct CPUs the game's threads were on (`timestamp, num_cores, cores`).
5.5 `process_tree.txt`
Every 10 s: a time stamp, then `ps --forest -o pid,ppid,nlwp,cputimes,comm` for the game's session (process ID, parent ID, number of threads, cumulative CPU seconds, command). It shows which processes belong to the game and how much CPU each has used, so child processes cannot be missed silently.
5.6 `cpu_log.csv`
Written by `cpu_log.py` (the same script as in Experiment 1) every 30 ms from `/proc/stat`, for the whole machine:
Column	Content
`timestamp`	Unix epoch seconds
`elapsed_s`	Seconds since the logger started
`cpu0` … `cpu15`	Busy % of that logical CPU over the last interval (100 × busy ticks / total ticks; idle includes iowait)
5.7 Console summary printed after every run
Demand summary: the p50/p95/max lines from §5.2.
Process-tree check: the top five processes in the session by CPU seconds, with their share. If more than one process uses at least 5 % of the total, a warning says that `pidstat` watches only one of them. If one process does essentially all the work, the script reports that `pidstat` is measuring the right thing. This check does not affect `demand.csv`, which covers the whole session either way.
---
6. Scripts used in Experiment 2
`run_all.py` — controller
The main operator. For each game and iteration it waits for the temperature target, starts `cpu_log.py`, runs the game launcher, stops `cpu_log.py`, keeps the three files described above, deletes the rest, and prints the summaries. It converts SIGTERM and SIGHUP into a normal exit so `cpu_log.py` is still stopped if the terminal is closed.
`run_game_cpu_core.py` — one game, one timed run with CPU samplers
A variant of `run_game.py`. It sets up MangoHud and PRIME offload, launches the game in a new session, starts the four samplers (`demand_profile.py`, the process-tree logger, `nvidia-smi`, `pidstat`), stops the game after the duration (SIGINT, then SIGKILL), and does the final pidstat parsing. MangoHud and `nvidia-smi` still run here, but their output is discarded by `run_all.py`.
`demand_profile.py` — cores-worth sampler
Samples a whole process session every 0.25 s and writes `demand.csv`. It can also print a summary of an existing file with `--summary`. P-core and E-core CPU lists are set by `--p-cpus` (default `0-11`) and `--e-cpus` (default `12-15`); these match the test laptop (Appendix A). `run_game_cpu_core.py` launches it without those options, so on a different CPU the defaults inside `demand_profile.py` would have to be edited.
`cpu_log.py` — per-core utilisation
Unchanged from Experiment 1: per-core busy % from `/proc/stat`, every 30 ms by default.
---
7. How to read the data
Question	Where to look
How much CPU does the game need?	`demand.csv` `total_cores`: p50 for the typical load, p95 for near-peak demand, max for the worst moment
Does it run on P-cores or E-cores?	`p_cores` vs `e_cores`
Is it limited by one thread?	`busiest_thread_cores`. A value near 1.0 means one thread is saturating a logical CPU
How many threads and processes are involved?	`n_threads`, `n_procs`, and `process_tree.txt`
Which logical CPUs did the game's threads land on?	`pidstat_raw.txt` (after conversion)
What was the whole machine doing, per core, at fine resolution?	`cpu_log.csv`, aligned with `demand.csv` via `timestamp` / `epoch`
Points to keep in mind when interpreting results:
Cores-worth counts logical CPUs. The P-cores have two threads each (SMT) that share one physical core, so a cores-worth of 2.0 spread over two sibling threads is not the same as two full physical cores.
Frame caps shape the result. Luanti (60 fps) and SuperTuxKart (144 fps, vsync on) are limited by their caps, while Xonotic and glmark2 are uncapped and will use whatever CPU they can.
P/E attribution is approximate. Each thread's CPU time in an interval is credited to the core type it was on at the end of that interval.
Sampling limits. Kernel CPU time ticks are 10 ms, so intervals below about 0.2 s are noisy. Processes that start and exit between two 1 s session scans are not seen.
The first 30 s are dropped by the printed summary, as warm-up. Compute your own statistics from `demand.csv` if you want a different window.
Compare with Experiment 1 with care. Experiment 2 has no AI job running, uses padding, and holds the CPU in the `performance` governor (Appendix A).
---
8. Practical notes for a third party repeating the run
Put `run_all.py`, `run_game_cpu_core.py`, `demand_profile.py` and `cpu_log.py` in one folder, and start `run_all.py` from a folder where the game paths in `run_game_cpu_core.py` resolve.
Edit the game paths in `run_game_cpu_core.py` (not `run_game.py`) if your install locations differ. Luanti's path is hardcoded to the test machine.
Use a working directory that does not hold Experiment 1 staging files. `run_all.py` deletes `./mangohud_raw/<game>/` before and after every run, and Experiment 1 uses the same folder name for its temporary files.
Re-running overwrites. Writing into an existing `cpu_runs/<game>/iter_XX/` replaces its files. Rename or move old results first (unlike Experiment 1, nothing is renamed automatically).
Check your CPU layout with `lscpu -e=CPU,CORE,SOCKET,MAXMHZ,MINMHZ` and edit the P/E defaults in `demand_profile.py` if it differs from Appendix A.
Install `sysstat` so that `pidstat` exists.
Do a short test first, for example `python run_all.py 60 -n 1 -g glmark2`, and confirm that all four files appear in `cpu_runs/glmark2/iter_01/`.
Keep the machine in the same state for all runs (power source, power mode, nothing else running). The setup used is listed in Appendix A.
---
---
Experiment 3 — Does separating CPU cores give performance back? (game and AI pinned to disjoint cores)
Experiment 3 asks a single question:
> If the game and the AI job are each confined to their own set of CPU cores with `taskset`, does the game get back the performance it lost to the AI in Experiment 1?
The setup is the Experiment 1 co-run (a game plus an AI job on the same machine), with one change: the game is pinned to a small set of cores and the AI job is pinned to the remaining cores, so the two can never compete for the same logical CPU.
---
1. What is being tested
1.1 Where this fits in the study
The CPU-partitioning study plan has these steps: confirm the topology (Step 1), measure each game's CPU demand alone (Step 2, Experiment 2), learn how to pin (Step 3), and run partition conditions (Step 4). Experiment 3 is the first partition test. It uses one partition per game, sized from the Experiment 2 demand data, and compares the result against the unpinned co-run of Experiment 1.
The sweep in the plan's Step 4 has further conditions that Experiment 3 does not include; they are planned next steps:
Plan condition	Status
U — unpinned game and AI	Experiment 1
A-m — game on m physical cores, AI on all the rest	Experiment 3, at one m per game
GP-m — game alone, pinned (control: does pinning itself hurt?)	Next step
B-m — game on m physical cores, AI on E-cores only	Next step
C — game on one SMT thread per core	Next step
Sweep over m = 1–4 (to find the knee)	Next step
Game-alone, unpinned baseline with FPS	Next step
Run-queue wait metric (`schedwait.py` in the plan)	Next step
Without the GP control, a change relative to Experiment 1 mixes two effects: the benefit of isolation and the cost of restricting the game to few cores.
1.2 Four games
The same four titles, resolutions and launch arguments as Experiments 1 and 2. Frame caps and vsync are as in Appendix A (Luanti 60 fps, SuperTuxKart 144 fps with vsync on, Xonotic and glmark2 uncapped). The game is forced onto the NVIDIA GPU by PRIME render offload, as before.
The difference is in `run_game.py` for this experiment: every game command starts with `taskset -c <cores>`, so the game and all of its threads and child processes inherit that mask (MangoHud runs around `taskset`, which then starts the game).
1.3 Two AI workloads, two placements, four intensities
Factor	Values
AI model	`cnn`, `tinybert` (Qwen is not run here)
Placement	`gpu` (model on the NVIDIA GPU), `cpu` (model on the CPU)
Intensity	`high`, `mid`, `low`, `idle` (the same sleep-after-each-request settings as Experiment 1 §1.4)
That is 2 × 2 × 4 = 16 combinations. The AI script is `run_ai_workload.py`, unchanged from Experiment 1.
Qwen is not included because its inference runs inside the separate Ollama server, which pinning the Python request script does not affect.
The AI process is pinned in both placements. With the model on the GPU, the CPU-side work (request loop, tokenisation, kernel launches) is still confined to the AI's cores; with the model on the CPU, the inference itself is.
1.4 CPU partitions
The core counts follow the plan's starting estimate: the game gets `ceil(p95 total cores-worth) + 1` logical CPUs, using the Experiment 2 demand data. The AI gets everything else.
Game	Game CPUs	What that covers	AI CPUs
Xonotic	`0-2`	P-core 0 (both threads) and one thread of P-core 1	`3-15`
glmark2	`0-1`	P-core 0 (both threads)	`2-15`
Luanti	`0-1`	P-core 0	`2-15`
SuperTuxKart	`0-1`	P-core 0	`2-15`
Logical CPUs 0–11 are P-core threads in sibling pairs (0,1), (2,3), …, (10,11), and 12–15 are E-cores (Appendix A.3).
The game masks are the `taskset` prefixes in each `GAMES_CONFIG` entry in this experiment's `run_game.py`.
The AI masks are set by `run_task.py` through `--ai-cores-xonotic` (default `3-15`) and `--ai-cores-default` (default `2-15`).
`taskset` sets which CPUs a process may use; it does not guarantee where threads run. The scheduler still chooses among the allowed CPUs.
1.5 Repeats and size
For each combination and each game, the run is repeated 5 times, each with a 240 s game window. A new AI process is started for every run.
Default sweep size: 16 combinations × 4 games × 5 runs = 320 runs.
Expected duration. `run_task.py` estimates each run as 120 s (AI alone) + 240 s (game) + 20 s overhead = 380 s, so the full sweep is at least 33 h 46 m. This excludes model loading and cooldown waits.
---
2. Temperature control
Before every run, with nothing of ours running, `run_task.py` waits until both the GPU and the CPU are at or below the target:
Setting	Default
`--gpu-target`	50 °C
`--cpu-target`	50 °C
`--cooldown-timeout`	600 s
`--poll-interval`	5 s
GPU temperature comes from `nvidia-smi`. CPU temperature comes from the `coretemp` or `k10temp` sensor in `/sys/class/hwmon` (preferring `Package id 0`, `Tctl` or `Tdie`).
`--no-cpu-wait` cools down on GPU temperature only.
If the target is not reached within the timeout, the run is still performed but marked `started_hot` in `meta.json`, and it is listed in `state.json` and in the final summary.
The starting temperatures are recorded in `meta.json` (`start_gpu_temp`, `start_cpu_temp`).
Unlike Experiment 1, the cooldown is a fixed 50 °C on both devices, not an adaptive target based on the AI's own temperature.
---
3. Inputs you must provide
3.1 Machine and software
Linux (per-core logging reads `/proc/stat`)
NVIDIA driver with `nvidia-smi`
MangoHud, `taskset` (util-linux) on `PATH`. `pkill` and `pgrep` (procps) are used to stop the game cleanly, with a fallback if they are missing.
A Python environment with `torch`, `torchvision`, `transformers`, `datasets` and `requests` for the AI script. If it lives in a virtual environment, point `--ai-python` at it.
CUDA available to that Python if any `gpu` combination is run (otherwise the AI script silently falls back to the CPU; the preflight check refuses to start in that case).
A CPU temperature sensor (`coretemp`/`k10temp`), unless `--no-cpu-wait` is used.
3.2 Files and working directory
Run `run_task.py` from the folder that contains this experiment's `run_game.py`. `run_task.py` imports `GAMES_CONFIG` and the MangoHud staging folder from it.
`cpu_log.py` must sit next to `run_task.py`. `run_ai_workload.py` is found through `--ai-script` (default `run_ai_workload.py` in the current folder); it is run from its own folder, so the CIFAR-10 `data/` directory stays next to it.
This `run_game.py` is not the same file as the Experiment 1 one. It differs by the `taskset` prefixes and by Xonotic's path (relative: `../xonotic/xonotic/all`). Keep the two in separate folders.
Luanti's path is still an absolute local path (`/run/media/tanish/...`) and would have to be changed on another machine.
`run_ai_workload.py` and `cpu_log.py` are identical to the Experiment 1 versions.
3.3 Command-line inputs
Full default sweep:
```bash
python run_task.py
```
Argument	Default	Meaning
`--runs`	5	Runs per game per combination
`--duration`	240	Game window in seconds
`--models`	cnn tinybert	AI models
`--intensities`	high mid low idle	Request intensities
`--devices`	gpu cpu	AI placement
`--games`	all four	Games to run
`--ai-cores-xonotic`	`3-15`	`taskset` cores for the AI while the game is Xonotic
`--ai-cores-default`	`2-15`	`taskset` cores for the AI with every other game
`--gpu-target` / `--cpu-target`	50 / 50	°C to reach before each run
`--no-cpu-wait`	off	Cool down on GPU temperature only
`--cooldown-timeout`	600	Max seconds for the cooldown
`--poll-interval`	5	Seconds between temperature checks
`--ai-solo-wait`	120	Seconds the AI runs alone before the game starts
`--ready-timeout`	300	Max seconds for the AI model to load
`--ai-script`	`run_ai_workload.py`	Path to the AI script
`--ai-python`	current Python	Interpreter used for the AI script
`--results-dir`	`result_task`	Where results are written
`--cpu-interval`	30	Per-core CPU sampling interval in ms
`--fresh`	off	Ignore saved progress and redo everything
`--dry-run`	off	Print the plan only
`--skip-preflight`	off	Skip the startup checks
Example of a reduced run:
```bash
python run_task.py --runs 3 --models cnn --intensities high mid --devices gpu
```
The preflight check verifies that `run_game.py` and `cpu_log.py` exist, `/proc/stat` is present, the AI script exists, `nvidia-smi`, `mangohud` and `taskset` are on `PATH`, the GPU (and CPU) temperature can be read, every game executable exists, and PyTorch can see CUDA if `gpu` combinations are requested.
---
4. What one run does, in order
For one (model, intensity, device) combination, one game and one run number `k`:
Create `result_task/<combo>/<game>/run_<k>/`. If that folder already exists from an interrupted attempt, rename it with a `_prev_<timestamp>` suffix so the files are not mixed.
Cool down until GPU and CPU are ≤ 50 °C (§2) and record the starting temperatures.
Start the AI job under `taskset` on the AI cores (`3-15` for Xonotic, `2-15` otherwise), writing `ai_workload.log` and `ai_requests.csv`.
Wait until the model has loaded, meaning `Starting <model>` appears in the AI log (up to `--ready-timeout`, 300 s).
AI runs alone for `--ai-solo-wait` (120 s), so the game always starts against an AI that is already running.
Start `cpu_log.py` (30 ms), writing `cpu_percore.csv`.
Run `run_game.py <game> 240 --padding 0`, which starts MangoHud and the pinned game, logs the NVIDIA GPU with `nvidia-smi`, and stops the game after 240 s. MangoHud logs for the first 230 s so its CSV is closed before the game is killed.
Stop `cpu_log.py` with SIGINT, so the per-core log covers only the game window.
Move the new MangoHud and `nvidia-smi` files from `mangohud_raw/<game>/` into the run folder.
Stop the AI job with SIGINT, so it prints its throughput and latency summary into `ai_workload.log`.
Write `meta.json` and record the run in `state.json`.
A run counts as failed if the AI exits early, never becomes ready, `run_game.py` exits non-zero, or no MangoHud CSV was produced. Partial data from a failed run is kept, the failure is recorded, and the sweep moves on.
If you interrupt with Ctrl+C, run the same command again: completed runs are skipped and failed ones are retried.
---
5. Outputs — everything that is saved
5.1 Directory layout
```text
result_task/
  state.json
  <model>_<intensity>_<device>/        e.g. cnn_high_gpu
    <game>/                            xonotic, glmark2, luanti, supertuxkart
      run_1/ … run_5/
        <mangohud>.csv                 game trace (frame time, FPS, CPU/GPU sensors)
        <mangohud>_summary.csv         MangoHud's own averages
        nvidia_smi_<timestamp>.csv     GPU trace from nvidia-smi (100 ms)
        ai_workload.log                AI stdout, with the final summary
        ai_requests.csv                one row per AI request
        cpu_percore.csv                per-core busy %, game window only
        meta.json                      description of the run
```
5.2 `state.json`
Progress file for the sweep: `completed` (run keys that finished), `failed` (run key → error), and `hot` (runs that started above the cooldown target).
5.3 `meta.json`
Key	Meaning
`combo`, `model`, `intensity`, `device`, `game`, `run`	The factors
`duration_s`	Game window (240)
`status`	`ok`, `failed`, `interrupted` or `incomplete`
`started`, `finished`	Local ISO timestamps
`started_hot`	True if the cooldown target was not reached in time
`start_gpu_temp`, `start_cpu_temp`, `gpu_target`, `cpu_target`	Temperatures at the start and the targets
`ai_cores`	The `taskset` mask given to the AI (the game's mask is not stored here; it is the `taskset` prefix in `run_game.py`)
`per_core_csv`, `cpu_interval_ms`	The per-core log and its interval
`game_started_epoch`, `game_finished_epoch`	Unix times of the game window, for aligning logs
`mangohud_csvs`	Number of MangoHud CSVs collected (summary and `nvidia_smi` files excluded)
`ai_requests`	`total_requests`, `fulfilled`, `failed`, parsed from the AI log
`error`	Set if the run failed
5.4 MangoHud and `nvidia-smi` files
Same configuration and columns as Experiment 1 (§5.5, §5.6): MangoHud at 100 ms with `log_duration` 230 s, and `nvidia-smi` sampled every 100 ms. These are the game-side results: frame times and FPS from MangoHud, and the discrete GPU's utilisation, temperature, power, clocks and memory from `nvidia-smi`.
5.5 `ai_workload.log` and `ai_requests.csv`
The log has the model-load messages, up to five request errors, and the final summary (total, fulfilled and failed requests, throughput, mean latency, min/p50/p95/p99/max). `ai_requests.csv` has one row per request:
Column	Content
`request_id`	1, 2, 3 …
`epoch_s`	Wall-clock start of the request, aligned with the other logs
`elapsed_s`	Seconds since the AI loop started
`latency_ms`	Request latency
`ok`	1 if it succeeded
`tokens`	Empty for `cnn` and `tinybert`
These files cover the AI-only period, the game window and the few seconds around it, because the AI runs for the whole run. Use `game_started_epoch` and `game_finished_epoch` to select the game window.
5.6 `cpu_percore.csv`
Written by `cpu_log.py` at 30 ms, starting just before the game and stopping right after it:
Column	Content
`timestamp`	Unix epoch seconds
`elapsed_s`	Seconds since the logger started
`cpu0` … `cpu15`	Busy % of each logical CPU over the last interval
This is the check that the partition held: load from the game should sit on its CPUs, and load from the AI on the AI's CPUs.
---
6. Scripts used in Experiment 3
`run_task.py` — controller
Builds the list of (model × intensity × device × game × run) items, runs the preflight, and performs each run as in §4. It stores progress in `state.json` for resuming, marks runs that began hot, and prints a summary of completed, hot and failed runs at the end. Unlike `run_batch.py` in Experiment 1, it starts a fresh AI process for every run.
`run_game.py` — one game, one timed run (pinned version)
The Experiment 1 launcher with `taskset -c …` added to each game command. It still starts MangoHud, the `nvidia-smi` logger and the game in a new process session, stops the game after the duration, and waits 3 s for the MangoHud flush. `run_task.py` calls it with `--padding 0`.
`run_ai_workload.py` and `cpu_log.py`
Identical to Experiment 1. `run_task.py` runs the AI script under `taskset`.
---
7. How to read the data
7.1 What each file answers
Question	Where to look
Did the game keep its frame rate and frame-time quality?	MangoHud CSV: mean FPS, 1 % low FPS, p99 frame time, share of frames over the frame-time target
Did the GPU behave differently?	`nvidia_smi_*.csv`: utilisation, clocks, power, temperature
What did the AI pay for the isolation?	`ai_requests.csv` and `ai_workload.log`: throughput and latency of fulfilled requests
Did the partition actually hold?	`cpu_percore.csv`: busy % on the game's CPUs vs the AI's CPUs
Did the run start under comparable conditions?	`meta.json`: `started_hot`, `start_gpu_temp`, `start_cpu_temp`
7.2 Method (from the study plan)
Analyse from the raw frame times, not per-second FPS plots.
Discard the first 60 s of each game run.
Report the mean and 95 % confidence interval across the 5 runs.
Game degradation = percentage drop in 1 % low FPS relative to the reference (for example Experiment 1's unpinned co-run with the same combination, or the unpinned game-alone baseline when it exists). Do the same for p99 frame time and the share of frames over the frame-time target.
AI cost = change in throughput and latency relative to the unpinned co-run.
Define "degradation removed" as: the pinned-with-AI confidence interval overlaps the pinned game-alone (GP) confidence interval. Without the GP control, an improvement over Experiment 1 cannot be attributed to isolation alone.
If degradation is gone with a clean split, scheduling contention was the cause. If some remains, the cause lies in shared resources that pinning does not separate (L3 cache, memory bandwidth, power and thermal budget, and the GPU for `gpu` placements).
7.3 Comparing with Experiment 1
The game launch arguments, the 240 s window, 0 s padding, the MangoHud settings, 5 runs per combination and the AI script are the same, so MangoHud results are directly comparable. Other conditions are not identical:
Factor	Experiment 1	Experiment 3
Game's CPU freedom	Any core	2–3 logical CPUs
AI's CPU freedom	Any core	Cores 2–15 (3–15 with Xonotic)
AI lifecycle	One AI process per combination, 20 game runs back to back	A fresh AI process for every run
Thermal state at game start	AI at steady state, about baseline + 2 °C	Cooled to 50 °C, then 120 s of AI alone
Cooldown	Waits while the AI keeps running, GPU only	Waits with nothing running, GPU and CPU
CPU log window	Whole combination	Game window only
AI models	`cnn`, `tinybert`, `qwen`	`cnn`, `tinybert`
Check `start_gpu_temp` and `start_cpu_temp` before attributing a difference to pinning, since a cooler start can raise boost clocks on its own. The `idle` intensity (model loaded, no requests) is the closest thing to a control inside this experiment. Frame caps limit what degradation can show for capped games (Luanti 60 fps, SuperTuxKart 144 fps).
---
8. Design choices and limitations
Xonotic's split goes through a physical core. The mask `0-2` / `3-15` puts logical CPU 2 on the game side and its SMT sibling, CPU 3, on the AI side, so the two can share P-core 1's execution units and caches. The scheduler tends to avoid a busy core's sibling when other CPUs are idle, so the overlap is more likely when the AI uses many threads (CPU-placed AI at high intensity). `cpu_percore.csv` shows it: look for `cpu3` busy at the same time as `cpu2`. The other three games never share a physical core with the AI. The plan's rule is to allocate whole physical cores. The AI-side mask can be changed with `--ai-cores-xonotic`, and the game-side mask in `run_game.py`.
Two logical CPUs are one physical core. `0-1` is the two SMT threads of P-core 0, so most games are confined to one physical core.
No AI thread cap. `run_task.py` applies `taskset` but does not set `OMP_NUM_THREADS`, `MKL_NUM_THREADS` or PyTorch's thread count, so a CPU-placed AI uses however many threads PyTorch chooses under the mask.
The game's mask is not saved per run. It is fixed in `run_game.py`; `meta.json` stores only the AI mask.
Intensity is a sleep, not a request rate. The AI sleeps after each request, so the achieved rate depends on latency. Use `ai_workload.log` and `ai_requests.csv` for the achieved throughput.
Isolation is only at the CPU-core level. The L3 cache, memory bandwidth, the package's power and thermal budget, and (for `gpu` placement) the GPU are still shared.
Some processes are not pinned. `run_task.py`, `cpu_log.py`, the `nvidia-smi` logger, `run_game.py`, the desktop compositor, interrupts and kernel threads can run on any core, including the game's.
No pinned game-alone control, no run-queue wait metric and no sweep over core counts yet (§1.1).
---
9. Practical notes for a third party repeating the run
Put this experiment's `run_game.py` (with `taskset`), `run_task.py` and `cpu_log.py` in one folder, separate from the Experiment 1 folder, and run `run_task.py` from there.
Change the Luanti path in `run_game.py` if your install differs. Xonotic's relative path must resolve from the folder you run in.
Check your CPU layout (Appendix A.3) before reusing the core masks. They assume 12 P-core threads (0–11) in sibling pairs and 4 E-cores (12–15).
Look at the plan first with `python run_task.py --dry-run`.
Do a short test run, for example `python run_task.py --runs 1 --games glmark2 --models cnn --intensities high --devices gpu --duration 60`, and check that `run_1/` contains a MangoHud CSV, an `nvidia_smi` CSV, `ai_workload.log`, `ai_requests.csv`, `cpu_percore.csv` and `meta.json`.
Verify the pinning on a real run: in `cpu_percore.csv`, load from the AI should sit on the AI's CPUs, and the game's load should sit on CPUs `0-1` (or `0-2`).
The sweep takes more than a day and a half. Use `--runs`, `--models`, `--intensities`, `--devices` and `--games` to cut it down, and run the most informative combinations first.
Keep the machine in the same state across runs (AC power, power mode, nothing else running). The setup used is in Appendix A.

---
Appendix A — Test machine and configuration (Experiments 2 and 3)
Everything below was recorded for the Experiment 2 runs, and Experiment 3 was run on the same machine and configuration.
A.1 System
Item	Value
Laptop CPU	Intel Core i5-13450HX (6 P-cores with 2 threads each, 4 E-cores)
Laptop GPU	NVIDIA RTX 4050 (plus the Intel integrated GPU, which is not used for the games)
Kernel	`7.0.0-38-generic` (`uname -r`)
NVIDIA driver	595.99.02
CUDA version (from `nvidia-smi`)	13.2
Power source	Plugged into AC for every run
Laptop power mode	Performance (LOQ performance mode)
Resting temperature before each run	50 °C
A.2 Commands run before the experiment
These set the CPU to a fixed performance mode and locked the GPU graphics clock:
```bash
sudo cpupower frequency-set -g performance      # CPU governor: performance
sudo nvidia-smi -lgc 2700,2700                  # lock GPU graphics clock to 2700 MHz (min,max)
```
A.3 CPU layout
16 logical CPUs on 10 physical cores: six P-cores with two threads each (SMT) and four single-thread E-cores.
Type	Logical CPUs	Physical cores	Threads per core	Max MHz	Min MHz
P-core	(0,1), (2,3), (4,5), (6,7), (8,9), (10,11)	6	2	4600	800
E-core	12, 13, 14, 15	4	1	3400	800
This matches the defaults in `demand_profile.py` (`--p-cpus 0-11`, `--e-cpus 12-15`). It was checked with:
```bash
lscpu -e=CPU,CORE,SOCKET,MAXMHZ,MINMHZ
for c in 0 2 4 6 8 10 12 13 14 15; do echo -n "cpu$c siblings: "; cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list; done
```
Reported thread siblings: `0-1`, `2-3`, `4-5`, `6-7`, `8-9`, `10-11` for the P-cores, and a single CPU each for 12, 13, 14 and 15.
A.4 Games
Game	Version	Resolution	Vsync	Frame cap
Xonotic	git build `bd6e300` (Wed Aug 5 01:17:12 2026 +1000)	1920×1080	off	uncapped
glmark2	2023.01	1920×1080	not present	uncapped
SuperTuxKart	1.5	1920×1080	on	144 fps
Luanti	5.18.0-dev-b4eb8b9-dirty (Linux)	window set by the game (launched with `--go --random-input`)	off	60 fps
Luanti build details: LuaJIT 2.1.1761786044 (OpenResty), built by GCC 15.2, running on Linux/7.0.0 x86_64, `BUILD_TYPE=Release`, `RUN_IN_PLACE=1`, `USE_CURL=1`, `USE_GETTEXT=1`, `USE_SOUND=1`, `STATIC_SHAREDIR="."`, `STATIC_LOCALEDIR="locale"`.
A.5 Run settings
Setting	Value
Duration per run	240 s
Padding before and after each run	10 s
Iterations per game	3
Cooldown target (CPU and GPU)	50 °C
Demand sampler interval	0.25 s
`pidstat` interval	1 s
`cpu_log.py` interval	30 ms
A.6 Experiment 3 run settings
Setting	Value
Game window per run	240 s (0 s padding)
Runs per game per combination	5
AI-alone period before the game	120 s
AI load timeout	300 s
Cooldown target (GPU and CPU)	50 °C, timeout 600 s
`cpu_log.py` interval	30 ms
Game CPUs	Xonotic `0-2`; glmark2, Luanti, SuperTuxKart `0-1`
AI CPUs	with Xonotic `3-15`; with the others `2-15`
