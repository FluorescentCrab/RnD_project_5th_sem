#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import time

# --- Game Executable Configurations ---
GAMES_CONFIG = {
    "xonotic": [
        "../xonotic/xonotic/all",
        "run",
        "sdl",
        "+vid_width",
        "1920",
        "+vid_height",
        "1080",
        "+nextdemo",
        "quit",
        "+playdemo",
        "demos/rnd_demo.dem",
        "+cl_maxfps",
        "0",
    ],
    "glmark2": ["glmark2", "--size", "1920x1080"],
    "luanti": [
        "/run/media/tanish/New Volume/RnD_project/luanti/bin/luanti",
        "--go",
        "--random-input",
    ],
    "supertuxkart": [
        "supertuxkart",
        "--demo-mode=1",
        "--demo-laps=100",
        "--demo-karts=8",
        "--screensize=1920x1080",
    ],
}

BASE_DIR = os.getcwd()
RAW_LOG_DIR = os.path.join(BASE_DIR, "mangohud_raw")
DEFAULT_PADDING_SEC = 10
DEFAULT_NVIDIA_PCI = "0000:01:00.0"  # from your nvidia-smi Bus-Id
os.makedirs(RAW_LOG_DIR, exist_ok=True)


def parse_pidstat(raw_path, csv_path):
    """Read `pidstat -t -p <pid>` output. Per timestamp, record the distinct
    CPU cores the game's threads were on and how many there are."""
    samples = []  # list of (timestamp, [cpu, ...])
    cur_ts, cur_rows = None, []
    with open(raw_path) as f:
        for line in f:
            tok = line.split()
            if len(tok) < 10 or tok[0].startswith("Average") or "UID" in tok:
                continue
            if tok[1] in ("AM", "PM"):  # 12h locale fallback
                tok = [tok[0] + " " + tok[1]] + tok[2:]
            if len(tok) < 10:
                continue
            ts, tid = tok[0], tok[3]
            if tid == "-":  # process summary row, keep only thread rows
                continue
            try:
                float(tok[8])
                cpu = int(tok[9])
            except ValueError:
                continue
            if ts != cur_ts:
                if cur_ts is not None:
                    samples.append((cur_ts, cur_rows))
                cur_ts, cur_rows = ts, []
            cur_rows.append(cpu)
    if cur_ts is not None:
        samples.append((cur_ts, cur_rows))

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "num_cores", "cores"])
        for ts, rows in samples:
            cores = sorted(set(rows))
            w.writerow([ts, len(cores), " ".join(map(str, cores))])
    counts = [len(set(r)) for _, r in samples]
    if counts:
        print(
            f"[+] pidstat: {len(samples)} samples, distinct cores "
            f"min/avg/max = {min(counts)}/{sum(counts)/len(counts):.1f}/{max(counts)}"
        )
    else:
        print(f"[!] No samples in {raw_path} ({os.path.getsize(raw_path)} bytes); see pidstat_err.txt")
    print(f"[+] Core usage CSV: {csv_path}")


def _nthreads(pid):
    try:
        return len(os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return 0


def _descendants(root):
    """All child/grandchild/... PIDs of root, read from /proc."""
    kids = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(ppid, []).append(int(name))
    out, stack = [], [root]
    while stack:
        for c in kids.get(stack.pop(), []):
            out.append(c)
            stack.append(c)
    return out


def find_game_pid(proc, timeout=60):
    """PID that pidstat should watch.
    Some games start through a wrapper shell script (e.g. Xonotic's `all`) that
    sits idle at 0% while the real game runs as its child. A wrapper has 1 thread,
    a real game has several, so: if the launched process is multi-threaded use it,
    otherwise wait for the descendant with the most threads."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _nthreads(proc.pid) > 1:
            return proc.pid
        if any(_nthreads(c) > 1 for c in _descendants(proc.pid)):
            time.sleep(2)  # let the game finish spawning its threads
            best = max(_descendants(proc.pid), key=_nthreads)
            return best
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    print("[!] Could not find a multi-threaded game process, watching the launched PID")
    return proc.pid


def main():
    parser = argparse.ArgumentParser(
        description="Run a single game benchmark using MangoHud."
    )
    parser.add_argument(
        "game_name",
        type=str,
        choices=list(GAMES_CONFIG.keys()),
        help="Name of the game to run",
    )
    parser.add_argument(
        "duration", type=int, help="Execution duration in seconds"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=DEFAULT_PADDING_SEC,
        help="Padding delay (seconds) before and after run",
    )
    parser.add_argument(
        "--pci-dev",
        type=str,
        default=None,
        help="Optional: pin MangoHud to one GPU, e.g. 0000:01:00.0 (can reduce HUD data)",
    )
    parser.add_argument(
        "--gpu-list",
        type=str,
        default="0",
        help='MangoHud gpu_list (default: 0). Use "" to disable',
    )
    parser.add_argument(
        "--pidstat-interval",
        type=int,
        default=1,
        help="pidstat sampling interval in seconds (default: 1)",
    )
    parser.add_argument(
        "--demand-dt",
        type=float,
        default=0.25,
        help="sampling interval (s) of demand_profile.py, the fine-grained CPU sampler (default: 0.25)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable MangoHud debug logging"
    )
    args = parser.parse_args()

    # PRIME render offload: "1" ENABLES offload to the NVIDIA GPU
    env = os.environ.copy()
    env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

    out_dir = os.path.join(RAW_LOG_DIR, args.game_name)
    os.makedirs(out_dir, exist_ok=True)

    # Log for less than the kill time so the CSV is flushed before termination
    log_duration = max(1, args.duration - 10)

    mango_opts = [
        "autostart_log=1",
        f"log_duration={log_duration}",
        "log_interval=100",
        f"output_folder={out_dir}",
        "cpu_stats",
        "gpu_stats",
        "cpu_temp",
        "gpu_temp",
        "cpu_power",
        "gpu_power",
        "gpu_core_clock",
        "gpu_mem_clock",
        "vram",
        "ram",
        "core_stats",
    ]
    if args.pci_dev:
        mango_opts.append(f"pci_dev={args.pci_dev}")
    if args.gpu_list:
        mango_opts.append(f"gpu_list={args.gpu_list}")
    env["MANGOHUD_CONFIG"] = ",".join(mango_opts)
    if args.debug:
        env["MANGOHUD_LOG_LEVEL"] = "debug"

    # Launch mangohud directly (no timeout binary, it breaks NVML sensor reading)
    game_cmd = ["mangohud"] + GAMES_CONFIG[args.game_name]

    if args.padding > 0:
        print(f"[*] Pre-run padding: waiting {args.padding}s...")
        time.sleep(args.padding)

    print(
        f"[*] Starting {args.game_name} (kill at {args.duration}s, MangoHud logging {log_duration}s)..."
    )

    # New session so we can kill the whole process tree
    game_proc = subprocess.Popen(game_cmd, env=env, start_new_session=True)

    print(f"the game pid {game_proc.pid }")

    # Fine-grained (0.25 s) CPU sampler over the WHOLE session (game + children + helper threads).
    # The game was started with start_new_session=True, so session id == game_proc.pid.
    demand_csv = os.path.join(out_dir, "demand.csv")
    demand_proc = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "demand_profile.py"),
         "--session", str(game_proc.pid), "-o", demand_csv, "--dt", str(args.demand_dt)],
        start_new_session=True,
    )

    # Sidecar process-tree logger: every 10 s, list everything in the game's session
    # with its cumulative CPU seconds, so child processes are never missed silently.
    tree_log = open(os.path.join(out_dir, "process_tree.txt"), "w", buffering=1)
    tree_proc = subprocess.Popen(
        ["bash", "-c",
         f"while true; do date +%T; ps --forest -o pid,ppid,nlwp,cputimes,comm -s {game_proc.pid}; echo; sleep 10; done"],
        stdout=tree_log, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    # Sidecar GPU logger: reads NVIDIA stats directly via NVML,
    # independent of MangoHud's GPU detection (its CSV may only log the iGPU)
    smi_csv = open(os.path.join(out_dir, "nvidia_smi.csv"), "w")
    smi_proc = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,temperature.gpu,power.draw,clocks.gr,clocks.mem,memory.used",
            "--format=csv",
            "-lms",
            "100",
        ],
        stdout=smi_csv,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Sidecar per-thread CPU logger (pidstat). ISO time => 24h timestamps.
    pidstat_raw_path = os.path.join(out_dir, "pidstat_raw.txt")
    pidstat_raw = open(pidstat_raw_path, "w", buffering=1)
    pidstat_env = env.copy()
    pidstat_env["LC_ALL"] = "C"
    pidstat_env["S_TIME_FORMAT"] = "ISO"
    pidstat_err = open(os.path.join(out_dir, "pidstat_err.txt"), "w")
    target_pid = find_game_pid(game_proc)
    try:
        with open(f"/proc/{target_pid}/comm") as f:
            target_name = f.read().strip()
    except OSError:
        target_name = "?"
    print(f"[*] pidstat target: pid {target_pid} ({target_name}); launched pid was {game_proc.pid}")
    pidstat_cmd = ["pidstat", "-t" ,"-u", "-p", str(target_pid), str(args.pidstat_interval)]
    if shutil.which("stdbuf"):  # line-buffer so output isn't lost
        pidstat_cmd = ["stdbuf", "-oL"] + pidstat_cmd
    print(f"[*] pidstat cmd: {' '.join(pidstat_cmd)}")
    pidstat_proc = subprocess.Popen(
        pidstat_cmd,
        stdout=pidstat_raw,
        stderr=pidstat_err,
        env=pidstat_env,
        start_new_session=True,
    )

    try:
        game_proc.wait(timeout=args.duration)
        
    except subprocess.TimeoutExpired:
        print(
            f"[*] Duration ({args.duration}s) reached, terminating {args.game_name}..."
        )
        try:
            os.killpg(game_proc.pid, signal.SIGINT)
            game_proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(game_proc.pid, signal.SIGKILL)
                game_proc.wait()
            except ProcessLookupError:
                pass

    # Give MangoHud a moment to flush the CSV
    time.sleep(3)

    # Stop the sidecar GPU logger
    smi_proc.terminate()
    try:
        smi_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        smi_proc.kill()
    smi_csv.close()

    # Stop pidstat with SIGINT so it flushes its output cleanly
    try:
        os.killpg(pidstat_proc.pid, signal.SIGINT)
        pidstat_proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pidstat_proc.kill()
    pidstat_raw.close()
    pidstat_err.close()

    # Stop the 0.25 s sampler (SIGINT -> it exits cleanly; the CSV is line-buffered anyway)
    try:
        os.killpg(demand_proc.pid, signal.SIGINT)
        demand_proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        demand_proc.kill()

    try:
        os.killpg(tree_proc.pid, signal.SIGTERM)
        tree_proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    tree_log.close()

    cores_csv = os.path.join(out_dir, "cpu_cores_used.csv")
    parse_pidstat(pidstat_raw_path, cores_csv)

    print(f"[+] Finished {args.game_name} run. Logs saved in {out_dir}")

    if args.padding > 0:
        print(f"[*] Post-run padding: waiting {args.padding}s...")
        time.sleep(args.padding)


if __name__ == "__main__":
    main()
