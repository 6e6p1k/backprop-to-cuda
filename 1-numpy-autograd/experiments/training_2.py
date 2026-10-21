from keras.datasets import mnist
import numpy as np
from architecture import Model, SGD, CrossEntropy, Linear, Softmax, Flatten, ReLU, Layer, xavier_uniform

def to_ohe(y, num_classes):
    ohe_y = np.zeros((y.size, num_classes), dtype=int)
    ohe_y[np.arange(y.size), y] = 1 
    return ohe_y


EPOCHS = 50
BATCH_SIZE = 256
NUM_CLASSES = 10

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_y = to_ohe(train_y, NUM_CLASSES)
x_batched = [train_X[i:i+BATCH_SIZE] for i in range(0, len(train_X), BATCH_SIZE)]
y_batched = [train_y[i:i+BATCH_SIZE] for i in range(0, len(train_y), BATCH_SIZE)]

custom_layer = Layer(
    forward = lambda x, w1, w2, b1, b2: ( (x @ w1 + b1).relu() @ w2 + b2 ).relu(),
    params = [
        ((28 * 28, 32), xavier_uniform),
        ((32, 64), xavier_uniform),
        ((1, 32), np.zeros),
        ((1, 64), np.zeros),
    ]
)

arch = Flatten() >> custom_layer >> Linear(64, 10) >> Softmax(-1)
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
        total_preds += pred.data.shape[0]
        right_preds += (np.argmax(pred.data, axis=-1) == np.argmax(y, axis=-1)).sum()
        losses.append(loss.data)
    
    avg_loss = np.mean(losses)
    print(f"epoch: {epoch + 1} | avg_acc: {right_preds / total_preds} | avg_loss: {avg_loss}")
