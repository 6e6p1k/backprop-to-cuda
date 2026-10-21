from parameter import Model, cross_entropy
from keras.datasets import mnist
import numpy as np

EPOCHS = 50
BATCH_SIZE = 256
NUM_CLASSES = 10

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X / 255.0

x_batched = [train_X[i:i+BATCH_SIZE] for i in range(0, len(train_X), BATCH_SIZE)]
y_batched = [train_y[i:i+BATCH_SIZE] for i in range(0, len(train_y), BATCH_SIZE)]

model = Model('Flatten() -> Linear(784, 32) -> Linear(32, 10) -> Softmax(-1)')

for epoch in range(EPOCHS):
    right_preds = 0
    total_preds = 0
    losses = []

    for x, y in zip(x_batched, y_batched):
        model.zero_grad()

        pred = model(x)
        loss = cross_entropy(pred, y, NUM_CLASSES, len(y))
        loss.backward()
        model.step('SGD', 1e-3)
        
        total_preds += pred.data.shape[0]
        right_preds += (np.argmax(pred.data, axis=-1) == y).sum()
        losses.append(loss.data)
    
    avg_loss = np.mean(losses)
    print(f"epoch: {epoch + 1} | avg_acc: {right_preds / total_preds} | avg_loss: {avg_loss}")
