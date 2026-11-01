#!/usr/bin/env python3
import argparse
import csv
import math
import time
import random
import requests
import torch
import torchvision
import torchvision.transforms as transforms
import os
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification

# --- Intensity Configurations ---
INTENSITY_CONFIG = {
    "high": 0.05,
    "mid": 0.5,
    "low": 2.0,
    "idle": None
}

QWEN_TIMEOUT = 120      # seconds before a qwen request counts as failed
MAX_ERRORS_PRINTED = 5  # don't spam the log if the server is down


def sync(device):
    """CUDA kernels are asynchronous; wait for them so the measured latency is the real one."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(sorted_vals, p):
    k = max(0, min(len(sorted_vals) - 1, math.ceil(p / 100 * len(sorted_vals)) - 1))
    return sorted_vals[k]


# ================= MODEL LOADERS & RUNNERS =================
def load_qwen():
    print("[*] Warming up Ollama...")
    # Connects to your default, already-running Ollama server
    url = "http://localhost:11434/api/generate"
    try:
        requests.post(url,
                      json={"model": "qwen:0.5b", "prompt": "hello", "stream": False}, timeout=60)
    except Exception as e:
        print(f"[!] Ollama warmup failed: {e}. Make sure Ollama is running.")

    print("[*] Loading Dolly-15k dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    return dataset["instruction"]

def run_qwen(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {"model": "qwen:0.5b", "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload, timeout=QWEN_TIMEOUT)
    response.raise_for_status()  # HTTP errors count as failed requests
    return response.json()

def load_cnn(device):
    print(f"[*] Loading ResNet20 and CIFAR-10 to {device}...")
    # Added trust_repo=True to prevent PyTorch hub from hanging/failing on newer versions
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, trust_repo=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)

    images = []
    for i, (img, _) in enumerate(loader):
        if i >= 100: break
        images.append(img)
    return model, images

def run_cnn(model, img, device):
    img = img.to(device)
    with torch.no_grad():
        out = model(img)
    sync(device)
    return out

def load_tinybert(device):
    print(f"[*] Loading TinyBERT and SST-2 to {device}...")
    model_name = "prajjwal1/bert-tiny"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # Removed force_download=True to prevent unnecessary network overhead/errors
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Reverted to "nyu-mll/glue" because newer datasets library versions require the full namespace/name format
    dataset = load_dataset("nyu-mll/glue", "sst2", split="validation")
    texts = dataset["sentence"][:1000]
    return model, tokenizer, texts

def run_tinybert(model, tokenizer, text, device):
    batch_texts = random.sample(text, min(32, len(text)))
    with torch.no_grad():
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
    sync(device)
    return out

# ================= MAIN BENCHMARK LOOP =================
def main():
    parser = argparse.ArgumentParser(description="Run AI model benchmark for a specific duration and intensity.")
    parser.add_argument("model_name", choices=["qwen", "cnn", "tinybert"], help="Model to benchmark")
    parser.add_argument("duration", type=int, help="Execution duration in seconds")
    parser.add_argument("intensity", choices=list(INTENSITY_CONFIG.keys()), help="Request intensity")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu", help="Device to run PyTorch models on (default: gpu)")
    parser.add_argument("--log-csv", default=None,
                        help="Per-request log CSV (default: ai_requests_<model>_<intensity>_<device>.csv in the current folder)")
    args = parser.parse_args()

    # Determine PyTorch device
    if args.device == "gpu" and not torch.cuda.is_available():
        print("[!] GPU requested but CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if args.device == "gpu" else "cpu")

    print(f"[*] Loading {args.model_name}...")
    if args.model_name == "qwen":
        model_data = load_qwen()
        if args.device == "cpu":
            print("[!] Note: Ollama uses its own background server. To force CPU, ensure Ollama was started with CUDA_VISIBLE_DEVICES=\"\"")
    elif args.model_name == "cnn":
        model_data = load_cnn(device)
    elif args.model_name == "tinybert":
        model_data = load_tinybert(device)

    log_path = args.log_csv or f"ai_requests_{args.model_name}_{args.intensity}_{args.device}.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    # epoch_s lines up with MangoHud/nvidia-smi timestamps; elapsed_s is time since the loop started
    writer.writerow(["request_id", "epoch_s", "elapsed_s", "latency_ms", "ok", "tokens"])
    log_file.flush()
    print(f"[*] Logging every request to {os.path.abspath(log_path)}")

    intensity_delay = INTENSITY_CONFIG[args.intensity]
    print(f"[*] Starting {args.model_name} (Device: {device}, Intensity: {args.intensity}, Duration: {args.duration}s)...")

    latencies = []   # successful requests only (seconds)
    start_time = time.time()
    req_count = 0    # attempted
    failed = 0

    try:
        while (time.time() - start_time) < args.duration:
            if args.intensity == "idle":
                time.sleep(1)
                continue

            req_epoch = time.time()
            req_start = time.perf_counter()
            ok, tokens = True, ""

            try:
                if args.model_name == "qwen":
                    prompt = random.choice(model_data)
                    resp = run_qwen(prompt)
                    tokens = resp.get("eval_count", "")  # tokens Ollama generated
                elif args.model_name == "cnn":
                    model, images = model_data
                    img = random.choice(images)
                    run_cnn(model, img, device)
                elif args.model_name == "tinybert":
                    model, tokenizer, texts = model_data
                    run_tinybert(model, tokenizer, texts, device)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                ok = False
                failed += 1
                if failed <= MAX_ERRORS_PRINTED:
                    print(f"[!] Request {req_count + 1} failed: {e}")

            latency = time.perf_counter() - req_start
            req_count += 1
            if ok:
                latencies.append(latency)
            writer.writerow([req_count, f"{req_epoch:.3f}", f"{req_epoch - start_time:.3f}",
                             f"{latency * 1000:.3f}", int(ok), tokens])
            log_file.flush()  # keep data even if we get killed

            if not ok:
                time.sleep(0.5)  # don't hammer a dead server
            if intensity_delay is not None and intensity_delay > 0:
                time.sleep(intensity_delay)

    except KeyboardInterrupt:
        print("\n[!] Benchmark interrupted by user.")
    finally:
        log_file.close()

    total_time = time.time() - start_time
    fulfilled = len(latencies)
    print(f"\n[+] Finished {args.model_name} benchmark on {device}.")
    print(f"Total Requests: {req_count}")
    print(f"Fulfilled: {fulfilled}")
    print(f"Failed: {failed}")

    if fulfilled > 0:
        s = sorted(latencies)
        print(f"Throughput: {fulfilled / total_time:.2f} fulfilled requests/sec")
        print(f"Avg Latency: {(sum(latencies) / fulfilled) * 1000:.2f} ms")
        print(f"Latency min/p50/p95/p99/max: {s[0] * 1000:.2f} / {percentile(s, 50) * 1000:.2f} / "
              f"{percentile(s, 95) * 1000:.2f} / {percentile(s, 99) * 1000:.2f} / {s[-1] * 1000:.2f} ms")
    else:
        print("No requests fulfilled (Idle mode or all failed).")
    print(f"Per-request log: {os.path.abspath(log_path)}")

if __name__ == "__main__":
    main()
