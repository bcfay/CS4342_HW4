import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = np.load("hw4_X.npy")
y = np.load("hw4_y.npy")
n = X.shape[1] // 2

x = np.arange(-8, +8, 0.01)
plt.scatter(X[0, 0:n], X[1, 0:n])
plt.scatter(X[0, n:], X[1, n:])

#graph_data = open('HW4_Vectors.csv', 'r').read()
df = pd.read_csv ('HW4_Vectors.csv')

#lines = df.split('\n')
ws = df.loc[:,"Weights"]
bs = df.loc[:,"Bases"]
#Rows of CSV:
#   row 0 = horizontal
#   row 1 = diagonal
#   row 2 = SVM
'''
for line in lines[1:]:
    if len(line) > 1:
        # batch_size, training_acc, testing_acc, training_time = line.split(',')
        data = line.split(',')
        
        weightList = data[1].split(" ")

        floats = [float(x) for x in weightList]
        ws.append(floats)
        bs.append(float(data[2]))
'''
print("Weights:", ws, "\nBases:", bs)

# Plot some arbitrary parallel lines (*not* separating hyperplanes) just for an example
#TODO make this based on ws and bs
# y = mx + b (linear equation)
# y = wx + b for us (?)
#yy = np.dot(np.atleast_2d(x).T, np.atleast_2d(np.array(ws))).T + np.array(bs)
#plt.plot(x.T, yy[1].T, 'k-')
w_test =  [[10000000005], [-5]]
#w_test = [[-0.3], [1]]
xvar = ws[0]
yvar = ws[1] #[0]
result = yvar.strip('][').split(' ')
yvar = float(result[0])
if(yvar == 0):
    yvar = 0.00000000001

#w_test_2 = [[np.dot(yvar, -1)], [xvar]]
t = np.dot(np.atleast_2d(x).T, np.atleast_2d(np.array(-1/yvar)).T).T
yy_test = t  #- 1.6
#plt.autoscale(enable=False, axis='y', tight=True)
plt.plot(x, yy_test.T, 'k--')
plt.grid()
#plt.plot(x, x * -1.9 + 3 + 1, 'k--')
#plt.plot(x, x * -1.9 + 3 - 1, 'k:')
plt.show()
