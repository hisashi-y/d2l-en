#Initializing Model Parameters

#%matplotpib inline
import random
import d2l
from mxnet import autograd, np, npx
from d2l import mxnet as d2l
npx.set_np()

w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()

def linreg(X, w, b): #@save
    """The linear regression model."""
    return np.dot(X, w) + b

def squared_loss(y_hat, y): #@save
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

def sgd(params, lr, batch_size): #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def data_iter(batch_size, features, labels): #@save
    num_examples = len(features)
    indices = list(range(num_examples))
    #The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

lr = 0.03
num_epochs = 30
batch_size = 10
net = linreg
loss = squared_loss

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y) #Minibatch loss in 'X' and 'y'
        #Becaue 'l' has a shape ('batch_size', 1) and is not a scalar
        #variable, the elements in 'l' are added together to obtain a new
        #variable, on which gradients with respect to ['w', 'b'] are computed
        l.backward()
        sgd([w, b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
