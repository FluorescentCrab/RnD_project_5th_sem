#!/usr/bin/env python3
"""Log per-core CPU utilization (%) to CSV by sampling /proc/stat.

Usage:
  ./cpu_log.py                      # 30 ms interval, writes cpu_log.csv, Ctrl-C to stop
  ./cpu_log.py -i 20 -d 60 -o out.csv
"""
import argparse
import signal
import time

running = True


def stop(*_):
    global running
    running = False


def read_cpus():
    """Return {cpu_index: (busy_ticks, total_ticks)} for each core."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                break
            name, *vals = line.split()
            if name == "cpu":  # aggregate line, skip
                continue
            v = list(map(int, vals[:8]))  # user nice system idle iowait irq softirq steal
            total = sum(v)
            idle = v[3] + v[4]  # idle + iowait
            out[int(name[3:])] = (total - idle, total)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interval", type=float, default=30, help="interval in ms (default 30)")
    ap.add_argument("-o", "--output", default="cpu_log.csv")
    ap.add_argument("-d", "--duration", type=float, default=0, help="seconds to run (0 = until Ctrl-C)")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    interval = args.interval / 1000.0
    prev = read_cpus()
    cores = sorted(prev)
    t0 = time.perf_counter()
    next_t = t0 + interval

    with open(args.output, "w", buffering=1 << 16) as f:
        f.write("timestamp,elapsed_s," + ",".join(f"cpu{c}" for c in cores) + "\n")
        while running:
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            now = time.perf_counter()
            cur = read_cpus()
            row = []
            for c in cores:
                db = cur[c][0] - prev[c][0]
                dt = cur[c][1] - prev[c][1]
                row.append(f"{100.0 * db / dt:.1f}" if dt > 0 else "0.0")
            f.write(f"{time.time():.3f},{now - t0:.3f}," + ",".join(row) + "\n")
            prev = cur
            next_t += interval  # fixed schedule: no cumulative drift
            if next_t < now:  # fell behind; resync instead of bursting
                next_t = now + interval
            if args.duration and now - t0 >= args.duration:
                break


if __name__ == "__main__":
    main()
