#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import time

# --- Game Executable Configurations ---
GAMES_CONFIG = {
    "xonotic": [
        "/run/media/tanish/New Volume/RnD_project/xonotic/xonotic/all",
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
    # Sidecar GPU logger: reads NVIDIA stats directly via NVML,
    # independent of MangoHud's GPU detection (its CSV may only log the iGPU)
    smi_csv = open(os.path.join(out_dir, f"nvidia_smi_{time.strftime('%Y%m%d_%H%M%S')}.csv"), "w")
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

    print(f"[+] Finished {args.game_name} run. Logs saved in {out_dir}")

    if args.padding > 0:
        print(f"[*] Post-run padding: waiting {args.padding}s...")
        time.sleep(args.padding)


if __name__ == "__main__":
    main()
