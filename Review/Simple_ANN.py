import math

def simple_ANN(x, w, t):
    total_e, e, y= 0, [], []
    for n in range(len(x)):
        v = 0
        for d in range(len(x[0])):
            v += x[n][d] * w[d]
        y.append(1/1+math.e**(-v)) # sigmoid function
        e.append(-t[n]*math.log(y[n])-(1-t[n])*math.log(1-y[n])) # BCE
    total_e = sum(e)/len(x)
    return (y, w, total_e)

def simple_ANN_Grad(x, w, t, iter, lr):
    total_e = 0
    for i in range(iter):
        e, y = [], []
        for n in range(len(x)):
            v = 0
            for d in range(len(x[0])):
                v += x[n][d] * w[d]
            y.append(1/1+math.e**(-v))  # sigmoid function
            e.append((y[n]-t[n])**2)    # MSE

            # Gradient descent to update weights
            for p in range(len(w)):
                d = 2*x[n][p]*(y[n]-t[n])*(1-y[n])*y[n]
                w[p] -= lr*d
        total_e = sum(e)/len(x)
        return (y, w, e)