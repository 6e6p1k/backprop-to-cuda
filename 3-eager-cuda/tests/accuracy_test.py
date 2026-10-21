import os
import torch
import numpy as np

os.add_dll_directory = getattr(os, "add_dll_directory", lambda x: None)

from bobrito import Tensor
from bobrito.bobr import bobr

# helpers
PASS = 0
FAIL = 0

def check(name, got: torch.Tensor, exp: torch.Tensor, atol=1e-3, rtol=1e-3):
    global PASS, FAIL
    got_np = got.cpu().float().numpy() if isinstance(got, torch.Tensor) else np.array(got, dtype=np.float32)
    exp_np = exp.cpu().float().numpy() if isinstance(exp, torch.Tensor) else np.array(exp, dtype=np.float32)
    if got_np.shape != exp_np.shape:
        print(f"[FAIL] {name}  shape mismatch: got {got_np.shape}, exp {exp_np.shape}")
        FAIL += 1
        return
    if np.allclose(got_np, exp_np, atol=atol, rtol=rtol):
        print(f"[PASS] {name}")
        PASS += 1
    else:
        diff = np.abs(got_np - exp_np)
        print(f"[FAIL] {name}  max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}")
        FAIL += 1

def b2t(b: Tensor) -> torch.Tensor:
    return torch.tensor(b.values, dtype=torch.float32).reshape(b.shape)

def t2b(t: torch.Tensor, device="cuda") -> Tensor:
    return Tensor(t.float().flatten().tolist(), list(t.shape), device=device)

def rand(*shape, device="cuda"):
    t = torch.randn(*shape)
    return t2b(t, device), t

# matmul
print("\nmatmul:")

ba, ta = rand(64, 128)
bb, tb = rand(128, 32)
check("matmul 2d", b2t(ba.matmul_fast(bb)), ta @ tb)

ba, ta = rand(8, 64, 128)
bb, tb = rand(8, 128, 32)
check("matmul batched", b2t(ba.matmul_fast(bb)), ta @ tb)

ba, ta = rand(4, 2, 32, 64)
bb, tb = rand(4, 2, 64, 16)
check("matmul batched 4d", b2t(ba.matmul_fast(bb)), ta @ tb)

ba, ta = rand(1, 1)
bb, tb = rand(1, 1)
check("matmul 1x1", b2t(ba.matmul_fast(bb)), ta @ tb)

# elementwise add
print("\nadd:")

ba, ta = rand(128, 256)
bb, tb = rand(128, 256)
check("add same shape", b2t(ba + bb), ta + tb)

# scalar broadcast
ba, ta = rand(64, 32)
bs = t2b(torch.tensor([2.5]), "cuda")
ts = torch.tensor([2.5])
check("add scalar broadcast", b2t(ba + bs), ta + ts)

# row broadcast  [64,32] + [1,32]
ba, ta = rand(64, 32)
bb, tb = rand(1, 32)
check("add row broadcast [64,32]+[1,32]", b2t(ba + bb), ta + tb)

# col broadcast [64,32] + [64,1], needs strides set right
ba, ta = rand(64, 32)
bb, tb = rand(64, 1)
check("add col broadcast [64,32]+[64,1]", b2t(ba + bb), ta + tb)

# elementwise mul
print("\nmul:")

ba, ta = rand(128, 256)
bb, tb = rand(128, 256)
check("mul same shape", b2t(ba * bb), ta * tb)

ba, ta = rand(64, 32)
bs = t2b(torch.tensor([3.0]), "cuda")
ts = torch.tensor([3.0])
check("mul scalar broadcast", b2t(ba * bs), ta * ts)

ba, ta = rand(64, 32)
bb, tb = rand(1, 32)
check("mul row broadcast", b2t(ba * bb), ta * tb)

ba, ta = rand(64, 32)
bb, tb = rand(64, 1)
check("mul col broadcast", b2t(ba * bb), ta * tb)

# sum
print("\nsum:")

ba, ta = rand(4, 8, 16)
check("sum dim=0", b2t(ba.sum(0)), ta.sum(0))
check("sum dim=1", b2t(ba.sum(1)), ta.sum(1))
check("sum dim=2", b2t(ba.sum(2)), ta.sum(2))
check("sum dim=0 keepdims", b2t(ba.sum(0, True)), ta.sum(0, keepdim=True))
check("sum dim=1 keepdims", b2t(ba.sum(1, True)), ta.sum(1, keepdim=True))
check("sum dim=2 keepdims", b2t(ba.sum(2, True)), ta.sum(2, keepdim=True))

# to_cuda / to_cpu roundtrip
print("\ndevice transfer:")

ba, ta = rand(32, 32, device="cpu")
bc = ba.to_cuda()
back = bc.to_cpu()
check("cpu->cuda->cpu roundtrip", b2t(back), ta)

ba, ta = rand(32, 32)   # starts on cuda
back = ba.to_cpu()
check("cuda->cpu", b2t(back), ta)

# chained ops
print("\nchained ops:")

ba, ta = rand(32, 64)
bb, tb = rand(64, 16)
bc, tc = rand(32, 16)

# (A @ B) + C
check("(A@B)+C", b2t(ba.matmul_fast(bb) + bc), (ta @ tb) + tc)

# (A * B).sum
ba, ta = rand(8, 16)
bb, tb = rand(8, 16)
check("(A*B).sum(1)", b2t((ba * bb).sum(1)), (ta * tb).sum(1))

# A + B * C
ba, ta = rand(16, 16)
bb, tb = rand(16, 16)
bc, tc = rand(16, 16)
check("A + B*C", b2t(ba + bb * bc), ta + tb * tc)

# numerical stability
print("\nnumerical edge cases:")

# large values
ba = t2b(torch.full((32, 32), 1000.0), "cuda")
ta = torch.full((32, 32), 1000.0)
bb = t2b(torch.full((32, 32), 1000.0), "cuda")
tb = torch.full((32, 32), 1000.0)
check("add large values", b2t(ba + bb), ta + tb)

# near-zero
ba = t2b(torch.full((32, 32), 1e-6), "cuda")
ta = torch.full((32, 32), 1e-6)
bb = t2b(torch.full((32, 32), 1e-6), "cuda")
tb = torch.full((32, 32), 1e-6)
check("mul near-zero", b2t(ba * bb), ta * tb, atol=1e-10, rtol=1e-3)

# zeros matmul
ba = t2b(torch.zeros(16, 16), "cuda")
ta = torch.zeros(16, 16)
check("matmul zeros", b2t(ba.matmul_fast(ba)), ta @ ta)

# identity matmul
eye = torch.eye(32)
ba = t2b(eye, "cuda")
bb, tb = rand(32, 32)
check("matmul identity", b2t(ba.matmul_fast(t2b(tb, "cuda"))), eye @ tb)

# summary
print(f"\n{'-'*50}")
print(f"  {PASS} passed  |  {FAIL} failed  |  {PASS+FAIL} total")
print(f"{'-'*50}")