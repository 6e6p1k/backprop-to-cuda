import os
import torch
from bobrito import Tensor
import time

def bench(fn, runs=300, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000  # ms

sizes = [
    ("tiny",   2,    3,    4),
    ("small",  32,   64,   128),
    ("medium", 128,  256,  512),
    ("large",  512,  1024, 512),
    ("1k",    1000, 1000, 1000),
    ("2^10",  1024, 1024, 1024),
    ("2k",    2000, 2000, 2000),
    ("2^11",  2048, 2048, 2048),
    ("3k",    3000, 3000, 3000),
    ("2^12",  4096, 4096, 4096),
]

print(f"{'':8} {'bobrito':>12} {'torch':>12} {'ratio':>10}")
print("-" * 46)

for name, M, K, N in sizes:
    # bobrito, tensors stay on gpu
    a = Tensor([1.0] * (M * K), [M, K], "cuda")
    b = Tensor([1.0] * (K * N), [K, N], "cuda")
    bobrito_ms = bench(lambda: a.matmul_fast(b))

    # torch
    ta = torch.ones(M, K, device="cuda")
    tb = torch.ones(K, N, device="cuda")
    torch_ms = bench(lambda: ta @ tb)

    ratio = bobrito_ms / torch_ms
    print(f"{name:8} {bobrito_ms:>11.4f}ms {torch_ms:>11.4f}ms {ratio:>9.2f}x")