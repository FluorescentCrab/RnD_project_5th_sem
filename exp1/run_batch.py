#!/usr/bin/env python3
"""
Controller: runs EVERY AI-workload combination and benchmarks the games under each one.
It does very little itself -- it just calls your existing scripts in the right order:

    run_ai_workload.py   -> the AI load (started in the background, stopped with SIGINT)
    run_game_batch.py    -> the game loop + cooldowns between runs (called once per combination)

For each combination (AI model x intensity x device):
    start AI -> warmup -> measure baseline GPU temp (avg) -> run_game_batch.py <runs> <duration>
    --target-temp <baseline+margin> -> stop AI -> collect CSVs into results/ -> cool down

Just run:   python run_batch.py
Defaults:   5 batches per combination, 240s (4 min) per game run.

Which games run is decided by GAMES_CONFIG in run_game.py (edit it there).
How a game run works is decided by run_game.py / run_game_batch.py (edit them there).

Results: results/<model>_<intensity>_<device>/<game>/<mangohud csv files>
         + meta.json and ai_workload.log per combination, and results/state.json for resuming.

Per-core CPU utilization: cpu_log.py (same folder as this script) is started right before each
combination's games and stopped right after them. One CSV per combination goes to a SEPARATE
folder:  per_core_result/<combo>.csv   (sampled every --cpu-interval ms, default 30).

Layout: run from the folder holding run_game.py / run_game_batch.py (next to your game data).
The AI script can live elsewhere: --ai-script ~/Desktop/run_ai_workload.py [--ai-python ~/Desktop/.venv/bin/python]
"""
import argparse
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

from run_game import GAMES_CONFIG, RAW_LOG_DIR            # game list + where MangoHud CSVs land
from run_game_batch import get_gpu_temp, wait_for_gpu_temp  # temperature helpers (single source of truth)

ALL_MODELS = ["cnn", "tinybert","qwen"]# 
ALL_INTENSITIES = ["high", "mid","low","idle"] #" 
ALL_DEVICES = ["gpu", "cpu"]

CPU_LOG_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_log.py")
AI_MAX_DURATION = 24 * 3600  # effectively "forever"; we stop the AI workload ourselves with SIGINT
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen:0.5b"


class ComboError(Exception):
    """A combination failed (AI workload died, never became ready, ...). Move on to the next one."""


def restore_sigint():
    # If this script was launched with SIGINT ignored (background job in a shell script,
    # some launchers), children would inherit that and ignore our clean-stop signal.
    signal.signal(signal.SIGINT, signal.SIG_DFL)


# ---------------------------------------------------------------- planning

def build_combos(models, intensities, devices):
    """qwen runs on Ollama's own server (the --device flag has no effect), so it gets one 'ollama' device."""
    combos = []
    for model in models:
        devs = ["ollama"] if model == "qwen" else devices
        for intensity in intensities:
            for dev in devs:
                combos.append((model, intensity, dev))
    return combos


def combo_name(c):
    return f"{c[0]}_{c[1]}_{c[2]}"


def fmt_hms(sec):
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def print_plan(all_combos, pending, args):
    games = list(GAMES_CONFIG)
    per_combo = args.warmup + args.baseline_duration + args.runs * len(games) * (args.duration + 10)
    print("=" * 60)
    print(f"[*] {len(all_combos)} combinations ({len(all_combos) - len(pending)} already done, {len(pending)} to run)")
    print(f"[*] Games (from run_game.py): {', '.join(games)}")
    print(f"[*] Per combination: {args.runs} batches x {len(games)} games x {args.duration}s runs")
    print(f"[*] Estimated time: ~{fmt_hms(per_combo)} per combination, "
          f">= {fmt_hms(per_combo * len(pending))} total (cooldown waits not included)")
    for c in pending:
        print(f"      - {combo_name(c)}")
    print("=" * 60)


# ---------------------------------------------------------------- state

def load_state(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            print(f"[!] Could not read {path}; starting with empty state.")
    return {"completed": [], "failed": {}}


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------- preflight

def preflight(combos, ai_script, ai_python):
    """Return a list of problems that would make the run fail hours in."""
    problems = []

    for f in ("run_game.py", "run_game_batch.py"):
        if not os.path.isfile(f):
            problems.append(f"{f} not found in {os.getcwd()} (run from the folder with the game scripts)")
    if not os.path.isfile(CPU_LOG_SCRIPT):
        problems.append(f"cpu_log.py not found next to run_batch.py ({CPU_LOG_SCRIPT})")
    if not os.path.exists("/proc/stat"):
        problems.append("/proc/stat not found (per-core CPU logging needs Linux)")
    if not os.path.isfile(os.path.expanduser(ai_script)):
        problems.append(f"AI script not found: {ai_script} (use --ai-script /path/to/run_ai_workload.py)")

    for tool in ("nvidia-smi", "mangohud", "timeout"):
        if shutil.which(tool) is None:
            problems.append(f"'{tool}' not found in PATH")
    if shutil.which("nvidia-smi") and get_gpu_temp() is None:
        problems.append("nvidia-smi is installed but the GPU temperature could not be read")

    for g, cmd in GAMES_CONFIG.items():
        exe = cmd[0]
        found = (os.path.isfile(exe) and os.access(exe, os.X_OK)) if os.sep in exe else shutil.which(exe)
        if not found:
            problems.append(f"game '{g}': executable not found: {exe}")

    # The AI script silently falls back to CPU if CUDA is missing, which would
    # make every "gpu" combination secretly a CPU run. Catch that now.
    if any(c[0] != "qwen" and c[2] == "gpu" for c in combos):
        r = subprocess.run([ai_python, "-c", "import torch; print(torch.cuda.is_available())"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            problems.append(f"could not import torch with {ai_python} (needed for cnn/tinybert; see --ai-python)")
        elif r.stdout.strip() != "True":
            problems.append("CUDA is not available to PyTorch, but 'gpu' combinations were requested "
                            "(they would silently run on CPU)")

    if any(c[0] == "qwen" for c in combos):
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
                names = [m.get("name", "") for m in json.load(resp).get("models", [])]
            if not any(n == OLLAMA_MODEL or n.startswith(OLLAMA_MODEL) for n in names):
                problems.append(f"Ollama is running but model '{OLLAMA_MODEL}' is not pulled "
                                f"(run: ollama pull {OLLAMA_MODEL})")
        except Exception as e:
            problems.append(f"Ollama not reachable at {OLLAMA_URL} ({e})")

    return problems


# ---------------------------------------------------------------- run_ai_workload.py (background)

def start_ai(combo, log_path, ai_script, ai_python):
    model, intensity, device = combo
    ai_script = os.path.abspath(os.path.expanduser(ai_script))
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [ai_python, "-u", ai_script, model, str(AI_MAX_DURATION), intensity,
         "--device", "gpu" if device == "ollama" else device],
        cwd=os.path.dirname(ai_script),  # its ./data (CIFAR-10) etc. stay next to the AI script
        stdout=log, stderr=subprocess.STDOUT,
        preexec_fn=restore_sigint,
    )
    print(f"[*] AI workload started (pid {proc.pid}): {model}, intensity={intensity}, device={device}")
    return proc, log


def stop_ai(proc, log):
    if proc.poll() is None:
        print("[*] Stopping AI workload ...")
        proc.send_signal(signal.SIGINT)  # run_ai_workload.py catches this and prints its stats
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("[!] AI workload didn't exit in 30s — killing it.")
            proc.kill()
            proc.wait()
    log.close()
    print("[+] AI workload stopped.")


# ---------------------------------------------------------------- cpu_log.py (background)

def start_cpu_log(combo, args):
    """Start cpu_log.py and wait until it is really sampling (CSV created) before returning."""
    out = os.path.join(args.per_core_dir, f"{combo_name(combo)}.csv")
    if os.path.exists(out):  # leftover from an earlier attempt: keep it, don't overwrite
        os.rename(out, f"{out[:-4]}_prev_{datetime.now():%Y%m%d_%H%M%S}.csv")
    proc = subprocess.Popen(
        [sys.executable, CPU_LOG_SCRIPT, "-i", str(args.cpu_interval), "-o", out],
        stdout=subprocess.DEVNULL,
        start_new_session=True,  # a terminal Ctrl+C reaches only us; we stop it ourselves, in order
    )
    deadline = time.time() + 5
    try:
        while not os.path.exists(out):
            if proc.poll() is not None:
                raise ComboError(f"cpu_log.py exited immediately (code {proc.returncode})")
            if time.time() > deadline:
                raise ComboError("cpu_log.py did not start sampling within 5s")
            time.sleep(0.05)
    except BaseException:  # ComboError or Ctrl+C: never leave the logger orphaned
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    print(f"[*] Per-core CPU logger started (pid {proc.pid}, {args.cpu_interval:g} ms) -> {out}")
    return proc, out


def stop_cpu_log(proc):
    if proc is None:
        return
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)  # cpu_log.py flushes and closes its CSV on SIGINT
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("[!] cpu_log.py didn't exit in 10s — killing it (tail of the CSV may be lost).")
            proc.kill()
            proc.wait()
        print("[+] Per-core CPU logger stopped.")
    else:
        print(f"[!] cpu_log.py had already exited (code {proc.returncode}); its CSV may be incomplete.")


def ai_check(proc, where):
    if proc.poll() is not None:
        raise ComboError(f"AI workload exited (code {proc.returncode}) {where}")


def wait_until_ready(proc, model, log_path, warmup, ready_timeout):
    """Fixed warmup, then confirm the AI script really got past model loading."""
    print(f"[*] Warmup: {warmup}s ...")
    end = time.time() + warmup
    while time.time() < end:
        ai_check(proc, "during warmup")
        time.sleep(max(0, min(5, end - time.time())))

    deadline = time.time() + ready_timeout
    while True:
        ai_check(proc, "while loading its model")
        with open(log_path) as f:
            if f"Starting {model}" in f.read():
                print("[+] AI workload is up and running requests.")
                return
        if time.time() > deadline:
            raise ComboError(f"AI workload still not ready {warmup + ready_timeout}s after start")
        time.sleep(2)


def measure_baseline(duration, poll_interval, proc):
    print(f"[*] Measuring baseline GPU temp for {duration}s ...")
    samples, start = [], time.time()
    while time.time() - start < duration:
        ai_check(proc, "during baseline measurement")
        temp = get_gpu_temp()
        if temp is not None:
            samples.append(temp)
            print(f"    [{int(time.time() - start):>4}s] {temp}°C")
        time.sleep(poll_interval)
    if not samples:
        raise ComboError("could not read any GPU temperatures for the baseline")
    avg = statistics.mean(samples)
    print(f"[+] Baseline avg: {avg:.2f}°C (min {min(samples)}, max {max(samples)}, {len(samples)} samples)")
    return avg


# ---------------------------------------------------------------- run_game_batch.py (blocking, watched)

def signal_session(sid, sig):
    """Send `sig` to EVERY process in the session `sid`.
    (`timeout` moves its child into its own process group, so killing just our group would miss the game.)"""
    try:
        subprocess.run(["pkill", f"-{int(sig)}", "-s", str(sid)], check=False)
    except FileNotFoundError:  # no pkill installed: fall back to the process group
        try:
            os.killpg(sid, sig)
        except ProcessLookupError:
            pass


def session_alive(sid):
    try:
        return subprocess.run(["pgrep", "-s", str(sid)], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def stop_game_batch(gp):
    """Ctrl+C-equivalent for the whole run_game_batch -> run_game -> timeout -> mangohud -> game tree."""
    sid = gp.pid  # started with start_new_session=True, so its pid is the session id
    if gp.poll() is None or session_alive(sid):
        print("[*] Stopping run_game_batch.py (and the game it is running) ...")
        signal_session(sid, signal.SIGINT)
        try:
            gp.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("[!] run_game_batch.py didn't exit in 30s — killing it.")
            signal_session(sid, signal.SIGKILL)
            gp.wait()
        for _ in range(10):  # let stragglers (the game itself) exit cleanly, then force
            if not session_alive(sid):
                return
            time.sleep(0.5)
        signal_session(sid, signal.SIGKILL)


def call_game_batch(args, target_temp, ai_proc):
    """Run `run_game_batch.py <runs> <duration> --target-temp ...` and watch the AI workload meanwhile."""
    cmd = [sys.executable, "run_game_batch.py", str(args.runs), str(args.duration),
           "--target-temp", str(target_temp),
           "--poll-interval", str(args.poll_interval),
           "--cooldown-timeout", str(args.cooldown_timeout)]
    print(f"[*] Calling: {' '.join(cmd)}")
    # own process group so we can stop the entire tree cleanly
    gp = subprocess.Popen(cmd, start_new_session=True, preexec_fn=restore_sigint)
    try:
        while gp.poll() is None:
            ai_check(ai_proc, "while the games were running")
            try:
                gp.wait(timeout=2)  # same 2s AI check, but returns the instant the batch ends
            except subprocess.TimeoutExpired:
                pass
        if gp.returncode != 0:
            raise ComboError(f"run_game_batch.py exited with code {gp.returncode}")
    finally:
        stop_game_batch(gp)  # no-op if it already finished


# ---------------------------------------------------------------- collecting results

def snapshot_raw():
    """Which files are already in mangohud_raw/<game>/ right now (so we can tell what a batch added)."""
    snap = {}
    for game in GAMES_CONFIG:
        d = os.path.join(RAW_LOG_DIR, game)
        os.makedirs(d, exist_ok=True)
        snap[game] = set(os.listdir(d))
    return snap


def collect_csvs(before, cdir):
    """Move the files created since `before` into cdir/<game>/. Returns {game: number_of_csvs}."""
    counts = {}
    for game, old in before.items():
        src = os.path.join(RAW_LOG_DIR, game)
        new = sorted(set(os.listdir(src)) - old)
        n = 0
        if new:
            dest = os.path.join(cdir, game)
            os.makedirs(dest, exist_ok=True)
            for fname in new:
                moved = os.path.join(dest, fname)
                shutil.move(os.path.join(src, fname), moved)
                if fname.endswith(".csv") and not fname.endswith("_summary.csv") and os.path.getsize(moved) > 0:
                    n += 1
        counts[game] = n
    return counts


# ---------------------------------------------------------------- one combination

def run_combo(combo, args):
    model, intensity, device = combo
    name = combo_name(combo)
    cdir = os.path.join(args.results_dir, name)
    if os.path.exists(cdir):  # leftover from an interrupted/failed attempt: keep it, don't mix
        os.rename(cdir, f"{cdir}_prev_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(cdir)
    log_path = os.path.join(cdir, "ai_workload.log")
    meta_path = os.path.join(cdir, "meta.json")
    meta = {"combo": name, "model": model, "intensity": intensity, "device": device,
            "runs_per_game": args.runs, "duration_s": args.duration,
            "started": datetime.now().isoformat(timespec="seconds")}

    before = snapshot_raw()
    proc, log = start_ai(combo, log_path, args.ai_script, args.ai_python)
    try:
        wait_until_ready(proc, model, log_path, args.warmup, args.ready_timeout)
        baseline = measure_baseline(args.baseline_duration, args.poll_interval, proc)
        # run_game_batch.py takes a whole-degree --target-temp, so round up
        target = math.ceil(baseline + args.temp_margin)
        meta.update(baseline_avg=round(baseline, 2), cooldown_target=target)
        save_json(meta_path, meta)
        print(f"[+] Cooldown target between runs: {target}°C (baseline {baseline:.2f} + {args.temp_margin}, rounded up)")

        cpu_proc, cpu_csv = start_cpu_log(combo, args)  # sampling starts right before the games ...
        try:
            meta.update(per_core_csv=cpu_csv, cpu_interval_ms=args.cpu_interval)
            save_json(meta_path, meta)
            call_game_batch(args, target, proc)
        finally:
            stop_cpu_log(cpu_proc)  # ... and ends right after them (also on failure / Ctrl+C)
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
    finally:
        counts = collect_csvs(before, cdir)  # runs even on failure/Ctrl+C so partial data is kept and sorted
        meta["csv_counts"] = counts
        stop_ai(proc, log)
        save_json(meta_path, meta)

    return {g: f"{c}/{args.runs}" for g, c in counts.items() if c < args.runs}


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Benchmark games under every AI-workload combination.")
    p.add_argument("--runs", type=int, default=5, help="Batches per combination (default: 5)")
    p.add_argument("--duration", type=int, default=240, help="Seconds per game run (default: 240 = 4 min)")
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    p.add_argument("--intensities", nargs="+", choices=ALL_INTENSITIES, default=ALL_INTENSITIES)
    p.add_argument("--devices", nargs="+", choices=ALL_DEVICES, default=ALL_DEVICES,
                   help="Devices for cnn/tinybert (qwen always uses Ollama's own server)")
    p.add_argument("--warmup", type=int, default=300, help="Seconds for the AI workload to start up (default: 300)")
    p.add_argument("--ready-timeout", type=int, default=120,
                   help="Extra seconds to wait if the model isn't loaded after warmup (default: 120)")
    p.add_argument("--baseline-duration", type=int, default=240, help="Seconds to measure baseline temp (default: 240)")
    p.add_argument("--temp-margin", type=float, default=2.0, help="°C added to baseline avg for cooldown target (default: 2)")
    p.add_argument("--idle-temp", type=float, default=45, help="GPU temp to reach between combinations (default: 45)")
    p.add_argument("--poll-interval", type=int, default=5)
    p.add_argument("--cooldown-timeout", type=int, default=600)
    p.add_argument("--ai-script", default="run_ai_workload.py",
                   help="Path to run_ai_workload.py (can be on a different drive than the game scripts)")
    p.add_argument("--ai-python", default=sys.executable,
                   help="Python used to run the AI script, e.g. a venv with torch (default: this Python)")
    p.add_argument("--results-dir", default="results",
                   help="Where results go (default: ./results; can be on your Ubuntu drive)")
    p.add_argument("--per-core-dir", default="per_core_result",
                   help="Separate folder for the per-core CPU utilization CSVs (default: ./per_core_result)")
    p.add_argument("--cpu-interval", type=float, default=30,
                   help="Per-core CPU sampling interval in ms (default: 30)")
    p.add_argument("--fresh", action="store_true", help="Ignore saved progress and redo every combination")
    p.add_argument("--dry-run", action="store_true", help="Only print the plan")
    p.add_argument("--skip-preflight", action="store_true", help="Skip the startup sanity checks")
    args = p.parse_args()

    all_combos = build_combos(args.models, args.intensities, args.devices)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.per_core_dir, exist_ok=True)
    state_path = os.path.join(args.results_dir, "state.json")
    state = {"completed": [], "failed": {}} if args.fresh else load_state(state_path)
    pending = [c for c in all_combos if combo_name(c) not in state["completed"]]

    print_plan(all_combos, pending, args)
    if args.dry_run or not pending:
        return

    if not args.skip_preflight:
        problems = preflight(pending, args.ai_script, args.ai_python)
        if problems:
            print("[!] Preflight checks failed:")
            for pr in problems:
                print(f"    - {pr}")
            sys.exit(1)
        print("[+] Preflight checks passed.")

    missing_report = {}
    try:
        for i, combo in enumerate(pending):
            name = combo_name(combo)
            print(f"\n{'#' * 60}\n[#] Combination {i + 1}/{len(pending)}: {name}\n{'#' * 60}")
            try:
                missing = run_combo(combo, args)
                state["completed"].append(name)
                state["failed"].pop(name, None)
                if missing:
                    missing_report[name] = missing
                    print(f"[!] {name} finished but some games produced fewer CSVs than expected: {missing}")
            except ComboError as e:
                state["failed"][name] = str(e)
                print(f"[!] {name} FAILED: {e}. Moving on to the next combination.")
            save_json(state_path, state)

            if i < len(pending) - 1:
                print("[*] Cooling down before the next combination ...")
                wait_for_gpu_temp(args.idle_temp, args.poll_interval, args.cooldown_timeout)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Re-run the same command to resume.")
        sys.exit(130)

    print("\n" + "=" * 60)
    print(f"[+] Done. Completed: {len(state['completed'])}/{len(all_combos)}")
    if state["failed"]:
        print("[!] Failed combinations (re-run the same command to retry them):")
        for n, why in state["failed"].items():
            print(f"    - {n}: {why}")
    for n, games in missing_report.items():
        print(f"[!] Fewer CSVs than expected in {n}: {games}")


if __name__ == "__main__":
    main()
