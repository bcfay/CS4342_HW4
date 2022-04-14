import numpy as np
import matplotlib.pyplot as plt

X = np.load("hw4_X.npy")
y = np.load("hw4_y.npy")
n = X.shape[1] // 2

x = np.arange(-8, +8, 0.01)
plt.scatter(X[0, 0:n], X[1, 0:n])
plt.scatter(X[0, n:], X[1, n:])

graph_data = open('HW3_part2.csv', 'r').read()
lines = graph_data.split('\n')
ws = []
bs = []

for line in lines[1:]:
    if len(line) > 1:
        # batch_size, training_acc, testing_acc, training_time = line.split(',')
        data = line.split(',')
        ws.append(float(data[0]))
        bs.append(float(data[1]))

print("Weights:", ws, "Bases:", bs)

# Plot some arbitrary parallel lines (*not* separating hyperplanes) just for an example
#TODO make this based on ws and bs
plt.plot(x, x * -1.9 + 3, 'k-')
plt.plot(x, x * -1.9 + 3 + 1, 'k--')
plt.plot(x, x * -1.9 + 3 - 1, 'k:')
plt.show()
