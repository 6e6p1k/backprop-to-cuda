import time
import numpy as np
from module import Model, SGD, CrossEntropy, Linear, LinearBlas, Softmax, Flatten

np.random.seed(42)

N          = 2048
EPOCHS     = 5
BATCH_SIZE = 256
IN         = 784
HIDDEN     = 256

x = np.random.randn(N, IN).astype(np.float32)
y = np.eye(10, dtype=np.float32)[np.random.randint(0, 10, N)]
x_batches = [x[i:i+BATCH_SIZE] for i in range(0, N, BATCH_SIZE)]
y_batches = [y[i:i+BATCH_SIZE] for i in range(0, N, BATCH_SIZE)]


def run(arch, label):
    model = Model(arch, optimizer=SGD(), loss=CrossEntropy())
    # warmup
    model.step(x_batches[0], y_batches[0], 1e-3)

    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        for xb, yb in zip(x_batches, y_batches):
            model.step(xb, yb, 1e-3)
    elapsed = time.perf_counter() - t0

    steps = EPOCHS * len(x_batches)
    print(f"{label:20s}  {elapsed:.3f}s total  ({elapsed/steps*1000:.1f} ms/step)")


naive = Flatten() >> Linear(IN, HIDDEN)     >> Linear(HIDDEN, HIDDEN) >> Linear(HIDDEN, 10) >> Softmax(-1)
blas  = Flatten() >> LinearBlas(IN, HIDDEN) >> LinearBlas(HIDDEN, HIDDEN) >> LinearBlas(HIDDEN, 10) >> Softmax(-1)

print(f"Network: {IN} -> {HIDDEN} -> {HIDDEN} -> 10   "
      f"| batch={BATCH_SIZE}  epochs={EPOCHS}  steps={EPOCHS * len(x_batches)}\n")
run(naive, "Naive matmul")
run(blas,  "BLAS matmul")
