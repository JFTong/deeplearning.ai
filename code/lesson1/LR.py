import numpy as np
from lr_utils import load_dataset

def sigmoid(z):
    s = 1.0 / (1 + np.exp(1 - np.array(z)))
    return s

def init_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = -1.0 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dZ = A - Y
    dw = 1.0 / m * np.dot(X, dZ.T)
    db = 1.0 / m * np.sum(dZ)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    grads = {'dw':dw,'db':db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db
        costs.append(cost)
        if print_cost == True and i % 100 == 0:
            print ("Cost after iterations %i:%f" %(i, cost))

    params = {"w":w,"b":b}
    grads = {"dw":dw,"db":db}
    return params, grads, cost
def predict(w, b, X):
    m = X.shape[1]
    y_predict = sigmoid(np.dot(w.T, X) + b)
    assert(y_predict.shape[1] == m)
    return y_predict

def model(X_train, Y_train, X_test, Y_test, num_iterations = 200, learning_rate = 0.5, print_cost = False):
    w, b = init_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    print Y_prediction_test
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d
if __name__ == "__main__":
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    print d
