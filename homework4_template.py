import pandas as pd

from cvxopt import solvers, matrix
import numpy as np
import sklearn.svm


class SVM4342:

    def __init__(self):
        pass

    # Expects each *row* to be an m-dimensional row vector. X should
    # contain n rows, where n is the number of examples.
    # y should correspondingly be an n-vector of labels (-1 or +1).
    def fit(self, X, y):
        print(X.shape)
        print(y.shape)

        sample_num, data_len = X.shape

        # ones = np.atleast_2d(np.ones(sample_num)).T
        X = self.data4wewights(X)
        bias_data_len = data_len + 1


        G = np.zeros((sample_num, bias_data_len))
        for i in range(bias_data_len):
            print(X[:, i])
            print(np.multiply(X[:, i], -y))
            G[:, i] = np.multiply(X[:, i], -y)  # makeynegativebysubtractingfromG
        # G = G.T

        h = np.ones(sample_num) * -1
        P = np.identity(bias_data_len)
        q = np.zeros(bias_data_len)

        # Solve -- if the variables above are defined correctly, you can call this as-is:
        sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))

        # Fetch the learned hyperplane and bias parameters out of sol['x']
        # To avoid any annoying errors due to broadcasting issues, I recommend
        # that you flatten() the w you retrieve from the solution vector so that
        # it becomes a 1-D np.array.
        flattened_soln = np.array(sol['x']).flatten()
        self.w = flattened_soln[:-1]  
        self.b = flattened_soln[-1]

    def data4wewights(self, X):
        sample_num, data_len = X.shape
        X = np.hstack((X, np.atleast_2d(np.ones(sample_num)).T))
        return X

    # Given a 2-D matrix of examples X, output a vector of predicted class labels
    def predict(self, x):
        samplenum, data_len_withbias = x.shape
        weights = np.atleast_2d(np.hstack((self.w, self.b))).T
        x = self.data4wewights(x)
        predictions = np.rint(np.dot(x, weights))
        for i in range(samplenum):
            if(predictions[i] > 1):
                predictions[i] = 1
            if(predictions[i] < -1):
                predictions[i] = -1
        return predictions

def remove_items(test_list, item):
    # using list comprehension to perform the task
    res = [i for i in test_list if i != item]

    return res

def test1():
    # Set up toy problem
    X = np.array([[1, 1], [2, 1], [1, 2], [2, 3], [1, 4], [2, 4]])
    y = np.array([-1, -1, -1, 1, 1, 1])

    # Train your model
    svm4342 = SVM4342()
    svm4342.fit(X, y)
    print(svm4342.w, svm4342.b)
    #TODO generate other two hyperplanes here
    ws = []
    bs = []
    #add vales for the first H -- this is normal to [0 1]T
    planeA = np.array([0, 1])
    planeA_x = planeA[1]
    planeA_y = planeA[0]
    if planeA_y == 0:
        planeA_y = 0.000000000001
    planeA_y = -1 / planeA_y

    normal_plane_w = np.array([planeA_y, planeA_x])   #TODO placeholder
    normal_plane_b = 0                              #TODO placeholder

    ws.append(normal_plane_w)
    bs.append(normal_plane_b)
   #add vales for the second H -- this is normal to [-0.3 1]T
    planeA2 = np.array([-0.3, 1])
    planeA_x2 = planeA2[1]
    planeA_y2 = planeA2[0]
    if planeA_y2 == 0:
        planeA_y2 = 0.000000000001
    second_plane_w = np.array( [planeA_y2, planeA_x2] ) #TODO placeholder
    second_plane_b = 0 #TODO placeholder

    ws.append(second_plane_w)
    bs.append(second_plane_b)
   #add vales for the third, trained H

    ws.append(svm4342.w)
    OFFSET = 0#1.89                            #TODO replace the offset with the actual bias of H
    bs.append(svm4342.b + OFFSET)

    vector2csv(ws, bs)

    # add vales for the fourth, trained H+
    yvar = ws[2]
    x_range = np.arange(-8, +8, 0.01)
    #result = yvar.strip('][').split(' ')
    result = remove_items(yvar, '')
    yvar = float(result[0])
    if (yvar == 0):
        yvar = 0.00000000001
    t = np.dot(np.atleast_2d(x_range).T, np.atleast_2d(np.array(-1 / yvar)).T).T

    # ACCOUNTING FOR THE CLOSEST ORANGE----------------------
    closestOrange = np.array([0, 0])
    #t_v2 = np.dot(np.atleast_2d(np.array(closestOrange[0])).T, np.atleast_2d(np.array(-1 / yvar)).T).T
    yy_test = t
    b_Hplus = closestOrange[1] - (closestOrange[0]*(-1)*ws[2][0] + svm4342.b)   # Y Index of the orange - Y index of the 3rd hyperplane (SVM)
    print("closest orange Y: ", closestOrange[1])
    print("redfunction Y at that x position: ", closestOrange[0]*(-1)*ws[2][0] + svm4342.b)
    print("H+ bias: ", b_Hplus)
    ws.append(svm4342.w)
    bs.append(svm4342.b + b_Hplus)
    vector2csv(ws, bs)


    #
    # add vales for the fifth, trained H-
    #yvar = ws[2]
    #x_range = np.arange(-8, +8, 0.01)
    # result = yvar.strip('][').split(' ')
    #result = remove_items(yvar, '')
    #yvar = float(result[0])
    #if (yvar == 0):
    #    yvar = 0.00000000001
    #t = np.dot(np.atleast_2d(x_range).T, np.atleast_2d(np.array(-1 / yvar)).T).T
    # ACCOUNTING FOR THE CLOSEST BLUE----------------------
    closestBlue = np.array([1, -5])
    # t_v2 = np.dot(np.atleast_2d(np.array(closestOrange[0])).T, np.atleast_2d(np.array(-1 / yvar)).T).T
    yy_test = t
    b_Hminus = closestBlue[1] - (closestBlue[0] * (-1)*ws[2][0] + svm4342.b)  # Y Index of the orange - Y index of the 3rd hyperplane (SVM) #flipped to negative intentionally - math did not work otherwise.
    print("closest blue Y: ", closestBlue[1])
    print("redfunction Y at that x position: ", closestBlue[0] * (-1)*ws[2][0] + svm4342.b)
    print("H- bias: ", b_Hminus)
    ws.append(svm4342.w)
    bs.append(svm4342.b + b_Hminus)

    # PROPERLY SETTING THE BIAS FOR H - IT NEEDS TO SET PRECISELY HALFWAY BETWEEN H+ AND H-  -----------
    avgBias =(bs[3] + bs[4]) / 2
    bs[2] = avgBias                       #bs[2] = SVM bias. this ensures the bias for H is perfectly between H+ and H-.
    vector2csv(ws, bs)
    print("\n\nMARGIN: ", bs[3] - bs[4])

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
    svm.fit(X, y)
    print(svm.coef_, svm.intercept_)

    our_pred = svm4342.predict(X).T
    print('our_pred', our_pred)
    lib_pred = svm.predict(X)
    print('lib_pred', lib_pred)

    acc = np.mean(our_pred == lib_pred)
    print("Acc={}".format(acc))


def test2(seed):
    np.random.seed(seed)

    # Generate random data
    X = np.random.rand(20, 3)
    # Generate random labels based on a random "ground-truth" hyperplane
    while True:
        w = np.random.rand(3)
        y = 2 * (X.dot(w) > 0.5) - 1
        # Keep generating ground-truth hyperplanes until we find one
        # that results in 2 classes
        if len(np.unique(y)) > 1:
            break

    svm4342 = SVM4342()
    svm4342.fit(X, y)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard margin
    svm.fit(X, y)
    print("svm coef shape = ");
    print(svm.coef_.flatten().shape)
    diff = np.linalg.norm(svm.coef_.flatten() - svm4342.w) + np.abs(svm.intercept_ - svm4342.b)
    print(diff)


    our_pred = svm4342.predict(X).T[0]
    print('our_pred', our_pred)
    lib_pred = svm.predict(X)
    print('lib_pred', lib_pred)

    acc = np.mean(our_pred == lib_pred)
    print("Acc={}".format(acc))

    if acc == 1 and diff < 1e-1:
        print("----------Passed!----------")

#w and b must be arrays
def vector2csv(w, b):
    file_path = "HW4_Vectors.csv"
    my_df = {'Weights': w,
             'Bases': b}
    df = pd.DataFrame(my_df)
    df.to_csv(file_path, header=True)


if __name__ == "__main__":
    test1()
    for seed in range(5):
        test2(seed)
