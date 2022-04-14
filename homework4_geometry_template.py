import numpy as np
import matplotlib.pyplot as plt

X = np.load("hw4_X.npy")
y = np.load("hw4_y.npy")
n = X.shape[1] // 2

x = np.arange(-8, +8, 0.01)
plt.scatter(X[0, 0:n], X[1, 0:n])
plt.scatter(X[0, n:], X[1, n:])

graph_data = open('HW4_Vectors.csv', 'r').read()
lines = graph_data.split('\n')
ws = []
bs = []

for line in lines[1:]:
    if len(line) > 1:
        # batch_size, training_acc, testing_acc, training_time = line.split(',')
        data = line.split(',')
        ws.append(float(data[1]))
        bs.append(float(data[2]))

print("Weights:", ws, "Bases:", bs)

# Plot some arbitrary parallel lines (*not* separating hyperplanes) just for an example
#TODO make this based on ws and bs
# y = mx + b (linear equation)
# y = wx + b for us (?)
#yy = np.dot(np.atleast_2d(x).T, np.atleast_2d(np.array(ws))).T + np.array(bs)
#plt.plot(x.T, yy[1].T, 'k-')
w_test =  [[0], [0]]
#w_test = [[-0.3], [1]]
xvar = w_test[0]
yvar = w_test[1]
w_test_2 = [[np.dot(yvar, -1)], [xvar]]
yy_test = np.dot(np.atleast_2d(x).T, np.atleast_2d(np.array(w_test)).T).T #- 1.6
plt.plot(x, yy_test.T, 'k--')
#plt.plot(x, x * -1.9 + 3 + 1, 'k--')
#plt.plot(x, x * -1.9 + 3 - 1, 'k:')
plt.show()
