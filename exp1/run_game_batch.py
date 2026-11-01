#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from run_game import GAMES_CONFIG


def get_gpu_temp():
    """Query the current GPU temperature using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        # Take the first GPU if multiple are present
        temp = int(result.stdout.strip().splitlines()[0])
        return temp
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError) as e:
        print(f"[!] Warning: Could not read GPU temperature ({e}).")
        return None


def wait_for_gpu_temp(target_temp, poll_interval=5, timeout=600):
    """Block until GPU temperature drops to target_temp or below."""
    print(f"[*] Waiting for GPU to cool down to ≤ {target_temp}°C ...")
    elapsed = 0
    while True:
        temp = get_gpu_temp()
        if temp is not None:
            if temp <= target_temp:
                print(f"[+] GPU temp is {temp}°C — ready to continue.")
                return
            else:
                print(f"    GPU temp: {temp}°C (target: ≤{target_temp}°C) — waiting {poll_interval}s ...")
        else:
            print(f"    Unable to read temp — waiting {poll_interval}s ...")

        time.sleep(poll_interval)
        elapsed += poll_interval

        if elapsed >= timeout:
            print(f"[!] Timeout ({timeout}s) reached. Continuing anyway.")
            return


def main():
    parser = argparse.ArgumentParser(description="Iterate through all games for batch benchmarking.")
    parser.add_argument("n", type=int, help="Number of iterations (times to run each game)")
    parser.add_argument("t", type=int, help="Duration of each run in seconds")
    parser.add_argument("--target-temp", type=int, default=45,
                        help="Target GPU temperature (°C) to wait for between runs (default: 45)")
    parser.add_argument("--poll-interval", type=int, default=5,
                        help="Seconds between GPU temp checks (default: 5)")
    parser.add_argument("--cooldown-timeout", type=int, default=600,
                        help="Max seconds to wait for cooldown before continuing anyway (default: 600)")
    args = parser.parse_args()

    games = list(GAMES_CONFIG.keys())
    total_runs = len(games) * args.n
    current_run = 0

    print(f"[*] Starting batch benchmark: {args.n} iterations, {args.t}s per run.")
    print(f"[*] Games to test: {', '.join(games)}")
    print(f"[*] Cooldown target: ≤{args.target_temp}°C between runs.")
    print("-" * 50)

    for i in range(args.n):
        print(f"\n[+] === Iteration {i + 1}/{args.n} ===")
        for game in games:
            current_run += 1
            print(f"\n[*] Running {game} ({current_run}/{total_runs})...")

            # Pass --padding 0 so run_game.py doesn't add its own fixed delay
            cmd = [
                sys.executable, "run_game.py",
                game,
                str(args.t),
                "--padding", "0"
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[!] Error running {game}: {e}")
            except KeyboardInterrupt:
                print("\n[!] Batch run interrupted by user.")
                sys.exit(1)

            # --- Cooldown wait (skip after the very last run) ---
            if current_run < total_runs:
                wait_for_gpu_temp(
                    target_temp=args.target_temp,
                    poll_interval=args.poll_interval,
                    timeout=args.cooldown_timeout
                )

    print("\n" + "=" * 50)
    print(f"[+] Batch benchmark complete! Ran {total_runs} total tests.")


if __name__ == "__main__":
    main()
