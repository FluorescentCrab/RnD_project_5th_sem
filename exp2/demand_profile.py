#!/usr/bin/env python3
"""CPU demand ("cores-worth") of a whole process session, sampled every DT seconds.

Usage:  demand_profile.py --session SID -o demand.csv [--dt 0.25]
        demand_profile.py --summary demand.csv          # p50/p95/max, first 30 s dropped

Samples utime+stime of every thread of every process in the session (so a game's
child processes and helper threads are included, no PIDs to type), from /proc.
cores-worth = CPU seconds used per wall-clock second.

Columns: epoch, t, total_cores, p_cores, e_cores, busiest_thread_cores, n_threads, n_procs
  epoch = wall-clock seconds (same clock as cpu_log.csv `timestamp`), t = seconds since start.
Notes: CPU time is attributed to the core type the thread was on at the end of each
interval (approximate). Kernel ticks are 10 ms, so dt below ~0.2 s is noisy.
Processes that start and exit between two 1-second session scans are not seen.
"""
import argparse
import csv
import os
import signal
import time

CLK = os.sysconf("SC_CLK_TCK")
running = True


def _stop(*_):
    global running
    running = False


def parse_cpus(spec):
    s = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        s.update(range(int(a), int(b or a) + 1))
    return s


def session_pids(sid):
    """PIDs whose session id is `sid` (the launcher started the game with its own session)."""
    pids = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat") as f:
                    rest = f.read().rsplit(")", 1)[1].split()
                if int(rest[3]) == sid:  # field 6 = session
                    pids.append(int(name))
            except (OSError, IndexError, ValueError):
                pass
    return pids


def snap(pids):
    """{(pid, tid): (cpu_ticks, cpu_core)} for every thread of every pid."""
    d = {}
    for pid in pids:
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for tid in tids:
            try:
                with open(f"/proc/{pid}/task/{tid}/stat") as f:
                    rest = f.read().rsplit(")", 1)[1].split()
                # field n is rest[n-3]: utime=14, stime=15, processor=39
                d[(pid, tid)] = (int(rest[11]) + int(rest[12]), int(rest[36]))
            except (OSError, IndexError, ValueError):
                pass
    return d


def _pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(csv_path, skip_s=30):
    """Return printable lines: p50/p95/max of total, P, E and busiest thread (first skip_s dropped)."""
    rows = list(csv.DictReader(open(csv_path)))
    kept = [r for r in rows if float(r["t"]) >= skip_s]
    note = f"first {skip_s}s dropped"
    if not kept:
        kept, note = rows, f"run shorter than {skip_s}s, nothing dropped"
    if not kept:
        return ["[!] demand.csv has no samples"]
    col = lambda k: [float(r[k]) for r in kept]
    tot, p, e, b = col("total_cores"), col("p_cores"), col("e_cores"), col("busiest_thread_cores")
    return [
        f"[demand] {len(kept)} samples ({note})  cores-worth p50/p95/max: "
        f"total {_pct(tot, 50):.2f}/{_pct(tot, 95):.2f}/{max(tot):.2f}",
        f"[demand] p95 P-core {_pct(p, 95):.2f} | p95 E-core {_pct(e, 95):.2f} | "
        f"p95 busiest thread {_pct(b, 95):.2f}",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int)
    ap.add_argument("-o", "--output", default="demand.csv")
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--p-cpus", default="0-11", help="logical CPUs that are P-cores (check lscpu)")
    ap.add_argument("--e-cpus", default="12-15", help="logical CPUs that are E-cores")
    ap.add_argument("--summary", metavar="CSV", help="just print the summary of an existing csv")
    a = ap.parse_args()
    if a.summary:
        print("\n".join(summarize(a.summary)))
        return
    if a.session is None:
        ap.error("--session is required")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    P, E = parse_cpus(a.p_cpus), parse_cpus(a.e_cpus)

    pids = session_pids(a.session)
    prev, t_prev = snap(pids), time.monotonic()
    t0 = t_prev
    next_t, next_scan = t0 + a.dt, t0 + 1.0
    with open(a.output, "w", newline="", buffering=1) as f:  # line-buffered: partial data survives a crash
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["epoch", "t", "total_cores", "p_cores", "e_cores",
                    "busiest_thread_cores", "n_threads", "n_procs"])
        while running:
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            now = time.monotonic()
            if now >= next_scan:                      # pick up child processes started later
                pids, next_scan = session_pids(a.session), now + 1.0
            cur = snap(pids)
            el = now - t_prev                         # real elapsed time, not the nominal dt
            tot = p = e = busiest = 0.0
            for key, (ticks, cpu) in cur.items():
                if key not in prev:
                    continue
                c = (ticks - prev[key][0]) / (CLK * el)
                tot += c
                busiest = max(busiest, c)
                if cpu in P:
                    p += c
                elif cpu in E:
                    e += c
            w.writerow([f"{time.time():.3f}", f"{now - t0:.2f}", f"{tot:.3f}", f"{p:.3f}",
                        f"{e:.3f}", f"{busiest:.3f}", len(cur), len(pids)])
            prev, t_prev = cur, now
            next_t += a.dt
            if next_t < now:                          # fell behind: resync, don't burst
                next_t = now + a.dt


if __name__ == "__main__":
    main()
