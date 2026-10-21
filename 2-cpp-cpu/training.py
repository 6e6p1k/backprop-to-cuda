from keras.datasets import mnist
import numpy as np
from module import Model, SGD, CrossEntropy, Linear, Softmax, Flatten, Layer, xavier_uniform, LinearBlas
from tools import rr
from tools import Tensor

def to_ohe(y, num_classes):
    ohe_y = np.zeros((y.size, num_classes), dtype=int)
    ohe_y[np.arange(y.size), y] = 1 
    return ohe_y


EPOCHS = 50
BATCH_SIZE = 256
NUM_CLASSES = 10

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.astype(np.float32) / 255.0

train_y = to_ohe(train_y, NUM_CLASSES)
x_batched = [train_X[i:i+BATCH_SIZE] for i in range(0, len(train_X), BATCH_SIZE)]
y_batched = [train_y[i:i+BATCH_SIZE] for i in range(0, len(train_y), BATCH_SIZE)]

arch = Flatten() >> LinearBlas(28 * 28, 64) >> LinearBlas(64, 10) >> Softmax(-1)
model = Model(
    architecture=arch,
    optimizer=SGD(),
    loss=CrossEntropy()
)


for epoch in range(EPOCHS):
    right_preds = 0
    total_preds = 0
    losses = []

    for x, y in zip(x_batched, y_batched):
        pred, loss = model.step(x, y, 1e-3)
        total_preds += pred.shape[0]
        right_preds += (rr.argmax(pred, dim=-1) == rr.argmax(rr.tsr(y), dim=-1)).sum().scalar
        losses.append(loss.scalar)
    avg_loss = rr.mean(losses)
    print(f"epoch: {epoch + 1} | avg_acc: {right_preds / total_preds:.4f} | avg_loss: {avg_loss.scalar:.4f}")
