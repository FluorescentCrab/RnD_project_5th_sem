#!/usr/bin/env python3
"""Run cpu_log.py and run_game_cpu_core.py together, N iterations per game.

Both existing scripts are called as-is (via subprocess); nothing is
reimplemented here. For every (game, iteration) it:
  0. waits until the CPU (and NVIDIA GPU, if present) has cooled to --temp-target
  1. starts cpu_log.py in the background  -> <run_dir>/cpu_log.csv
  2. runs   run_game_cpu_core.py <game> <duration>
  3. stops cpu_log.py (SIGINT, so it exits cleanly)
  4. copies ONLY pidstat_raw.txt from run_game_cpu_core's output into <run_dir>
     and removes the rest of its output (mangohud logs, nvidia_smi.csv, etc.),
     also keeps demand.csv (0.25 s cores-worth, whole session) and process_tree.txt (who ran in the game's session, with CPU seconds),
     (convert it to CSV later with the Jupyter cell)

Layout:
  cpu_runs/<game>/iter_01/cpu_log.csv
  cpu_runs/<game>/iter_01/pidstat_raw.txt
  cpu_runs/<game>/iter_01/demand.csv       (0.25 s cores-worth)
  cpu_runs/<game>/iter_01/process_tree.txt
  cpu_runs/<game>/iter_02/...

Usage (run from the same directory you normally run run_game_cpu_core.py from,
because its game paths are relative to the cwd):
  ./run_all.py 120 -n 5                          # all games, 5 iterations, 120 s each
  ./run_all.py 120 -n 3 -g xonotic glmark2       # only some games
  ./run_all.py 120 -n 3 --cpu-interval 20 --padding 5 -o my_runs
  ./run_all.py 120 -n 3 --temp-target 45 --temp-timeout 900   # cool to 45 C, wait max 15 min
  ./run_all.py 120 -n 3 --temp-target 0                       # disable temperature wait
"""
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time

from demand_profile import summarize as demand_summary

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES = ["xonotic", "glmark2", "luanti", "supertuxkart"]


_warned_no_sensor = set()


def read_cpu_temp():
    """CPU package temperature in C from hwmon/thermal, or None if not found."""
    best = None
    hw = "/sys/class/hwmon"
    if os.path.isdir(hw):
        for d in sorted(os.listdir(hw)):
            base = os.path.join(hw, d)
            try:
                name = open(os.path.join(base, "name")).read().strip()
            except OSError:
                continue
            if name not in ("coretemp", "k10temp", "zenpower", "cpu_thermal"):
                continue
            temps = {}
            for f in os.listdir(base):
                if f.startswith("temp") and f.endswith("_input"):
                    try:
                        val = int(open(os.path.join(base, f)).read()) / 1000.0
                        label_f = os.path.join(base, f.replace("_input", "_label"))
                        label = open(label_f).read().strip() if os.path.exists(label_f) else f
                        temps[label] = val
                    except (OSError, ValueError):
                        pass
            if not temps:
                continue
            for pref in ("Package id 0", "Tctl", "Tdie"):
                if pref in temps:
                    return temps[pref]
            best = max(temps.values())
    if best is not None:
        return best
    tz = "/sys/class/thermal"
    if os.path.isdir(tz):
        for d in sorted(os.listdir(tz)):
            try:
                if open(os.path.join(tz, d, "type")).read().strip() == "x86_pkg_temp":
                    return int(open(os.path.join(tz, d, "temp")).read()) / 1000.0
            except (OSError, ValueError):
                pass
    return None


def read_gpu_temp():
    """NVIDIA GPU temperature in C via nvidia-smi, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.split()
        return max(float(x) for x in out) if out else None
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def wait_for_cool(target, timeout, poll=5):
    """Block until CPU and GPU temps are <= target C (or timeout). target<=0 disables."""
    if target <= 0:
        return
    start = time.time()
    last_print = 0
    while True:
        sensors = {"CPU": read_cpu_temp(), "GPU": read_gpu_temp()}
        for k, v in sensors.items():
            if v is None and k not in _warned_no_sensor:
                _warned_no_sensor.add(k)
                print(f"[!] No {k} temperature sensor found; ignoring {k} for the wait")
        known = {k: v for k, v in sensors.items() if v is not None}
        if not known:
            return
        if all(v <= target for v in known.values()):
            print("[+] Cooled down: " + ", ".join(f"{k} {v:.0f}C" for k, v in known.items()))
            return
        waited = time.time() - start
        if timeout and waited >= timeout:
            print(f"[!] Temp wait timed out after {waited:.0f}s ("
                  + ", ".join(f"{k} {v:.0f}C" for k, v in known.items()) + "), continuing anyway")
            return
        if time.time() - last_print >= 15:
            last_print = time.time()
            print(f"[*] Waiting for temps <= {target:.0f}C: "
                  + ", ".join(f"{k} {v:.0f}C" for k, v in known.items())
                  + f" ({waited:.0f}s)")
        time.sleep(poll)


def stop_proc(proc, timeout=10):
    """Ask a child to stop with SIGINT; escalate if it doesn't."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def summarize_tree(path):
    """Print CPU seconds per process in the game's session (from process_tree.txt)."""
    best = {}  # pid -> [comm, max cumulative cpu seconds]
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return
    for line in lines:
        m = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[ \\_|-]*(\S.*)$", line)
        if m:
            pid, cpu_s, comm = int(m[1]), int(m[4]), m[5].strip()
            if pid not in best or cpu_s > best[pid][1]:
                best[pid] = [comm, cpu_s]
    total = sum(v[1] for v in best.values())
    if not total:
        print("[!] process_tree: no CPU time recorded")
        return
    ranked = sorted(best.items(), key=lambda kv: -kv[1][1])
    print("[tree] CPU seconds by process in the game's session:")
    for pid, (comm, cpu_s) in ranked[:5]:
        print(f"         {comm:<22} pid {pid:<7} {cpu_s:>6} s  {100 * cpu_s / total:>3.0f}%")
    heavy = [r for r in ranked if r[1][1] >= 0.05 * total]
    if len(heavy) > 1:
        print(f"[!] {len(heavy)} processes each use >=5% of the CPU: pidstat watches only one of them. "
              "Tell Claude which game this is.")
    else:
        print("[+] One process does essentially all the work, pidstat is measuring the right thing")


def run_once(args, game, it, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n--- {game} | iteration {it}/{args.iterations}: temperature check ---")
    wait_for_cool(args.temp_target, args.temp_timeout)
    cpu_csv = os.path.join(run_dir, "cpu_log.csv")

    # run_game_cpu_core.py writes to <cwd>/mangohud_raw/<game>/
    game_out = os.path.join(os.getcwd(), "mangohud_raw", game)
    shutil.rmtree(game_out, ignore_errors=True)  # no stale files from earlier runs

    cpu_cmd = [sys.executable, args.cpu_script, "-i", str(args.cpu_interval), "-o", cpu_csv]
    game_cmd = [
        sys.executable, args.game_script, game, str(args.duration),
        "--padding", str(args.padding),
        "--pidstat-interval", str(args.pidstat_interval),
    ]

    print(f"\n=== {game} | iteration {it}/{args.iterations} -> {run_dir} ===")
    # Own session so a Ctrl-C in this terminal doesn't hit it directly;
    # we stop it ourselves in the finally block.
    cpu_proc = subprocess.Popen(cpu_cmd, start_new_session=True)
    try:
        rc = subprocess.call(game_cmd)
        if rc != 0:
            print(f"[!] run_game_cpu_core.py exited with code {rc}")
    finally:
        stop_proc(cpu_proc)

    # Keep only the raw pidstat txt from the game script
    raw_src = os.path.join(game_out, "pidstat_raw.txt")
    if os.path.exists(raw_src):
        shutil.copy2(raw_src, os.path.join(run_dir, "pidstat_raw.txt"))
    else:
        print(f"[!] {raw_src} not found, nothing to save for this iteration")
    demand_src = os.path.join(game_out, "demand.csv")
    if os.path.exists(demand_src):
        shutil.copy2(demand_src, os.path.join(run_dir, "demand.csv"))
        for line in demand_summary(demand_src):
            print(line)
    else:
        print("[!] demand.csv not found")
    tree_src = os.path.join(game_out, "process_tree.txt")
    if os.path.exists(tree_src):
        shutil.copy2(tree_src, os.path.join(run_dir, "process_tree.txt"))
        summarize_tree(tree_src)
    shutil.rmtree(game_out, ignore_errors=True)

    print(f"[+] Saved: {cpu_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("duration", type=int, help="seconds per run (passed to run_game_cpu_core.py)")
    ap.add_argument("-n", "--iterations", type=int, default=3, help="iterations per game (default 3)")
    ap.add_argument("-g", "--games", nargs="+", choices=GAMES, default=GAMES, help="games to run (default: all)")
    ap.add_argument("-o", "--output", default="cpu_runs", help="output root folder (default: cpu_runs)")
    ap.add_argument("--cpu-interval", type=float, default=30, help="cpu_log.py interval in ms (default 30)")
    ap.add_argument("--padding", type=int, default=10, help="pre/post padding passed to run_game_cpu_core.py")
    ap.add_argument("--pidstat-interval", type=int, default=1, help="pidstat interval in s")
    ap.add_argument("--temp-target", type=float, default=50,
                    help="wait until CPU/GPU temp <= this many C before each run (default 50, 0 = disable)")
    ap.add_argument("--temp-timeout", type=int, default=600,
                    help="max seconds to wait for cooldown per run (default 600, 0 = no limit)")
    ap.add_argument("--cpu-script", default=os.path.join(HERE, "cpu_log.py"))
    ap.add_argument("--game-script", default=os.path.join(HERE, "run_game_cpu_core.py"))
    args = ap.parse_args()

    out_root = os.path.abspath(args.output)
    for game in args.games:
        for it in range(1, args.iterations + 1):
            run_dir = os.path.join(out_root, game, f"iter_{it:02d}")
            run_once(args, game, it, run_dir)
            time.sleep(2)  # brief settle between runs

    print(f"\n[+] All done. Results in {out_root}")


def _on_sigterm(*_):
    # Turn SIGTERM/SIGHUP (kill, closing the terminal) into a normal exit so the
    # `finally` in run_once still stops cpu_log.py instead of orphaning it.
    sys.exit(143)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGHUP, _on_sigterm)
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted")
