import time
import numpy as np
import torch
from bobrito import bobr

RUNS = 20
RESULTS_FILE = "./benchmark_results.txt"

def bench(fn, runs=RUNS):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.mean(times), np.std(times)

def cuda_vs_torch():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # (label, shapes)
    test_cases = [
        ("tiny   [2x3] @ [3x4]",       (2, 3), (3, 4)),
        ("small  [32x64] @ [64x128]",   (32, 64), (64, 128)),
        ("medium [128x256] @ [256x512]",(128, 256), (256, 512)),
        ("large  [512x1024] @ [1024x512]", (512, 1024), (1024, 512)),
        ("square [1000x1000] @ [1000x1000]", (1000, 1000), (1000, 1000)),
        ("giant  [10000x10000] @ [10000x10000]", (10000, 10000), (10000, 10000)),
    ]

    lines = []
    lines.append(f"MatMul Benchmark - {RUNS} runs each")
    lines.append(f"Torch device: {device}")
    lines.append("=" * 72)

    for label, (m, k), (k2, n) in test_cases:
        assert k == k2

        a_bobr = bobr.randn(m, k)
        b_bobr = bobr.randn(k, n)
        a_np = a_bobr.numpy()
        b_np = b_bobr.numpy()
        a_torch = torch.from_numpy(a_np).to(device)
        b_torch = torch.from_numpy(b_np).to(device)

        # warmup torch
        _ = a_torch @ b_torch
        if device == "cuda":
            torch.cuda.synchronize()

        cuda_mean, cuda_std = bench(lambda: a_bobr.matmul_cuda(b_bobr))

        if device == "cuda":
            def torch_fn():
                _ = a_torch @ b_torch
                torch.cuda.synchronize()
            torch_mean, torch_std = bench(torch_fn)
        else:
            torch_mean, torch_std = bench(lambda: a_torch @ b_torch)

        lines.append(f"\n{label}")
        lines.append(f"  cuda           :  {cuda_mean*1e3:10.4f} ms  ± {cuda_std*1e3:.4f} ms")
        lines.append(f"  torch({device}):  {torch_mean*1e3:10.4f} ms  ± {torch_std*1e3:.4f} ms")

        cuda_vs_torch  = cuda_mean / torch_mean if torch_mean > 0 else float('inf')
        lines.append(f"  [cuda vs torch: {cuda_vs_torch:.2f}x slower]")

    lines.append("\n" + "=" * 72)
    output = "\n".join(lines)
    print(output)

    with open(RESULTS_FILE, "w") as f:
        f.write(output + "\n")

    print(f"\nResults written to {RESULTS_FILE}")

cuda_vs_torch()