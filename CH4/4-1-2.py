#4.1.2 Activation Functions
from mxnet import autograd, np, npx
from d2l import mxnet as d2l

npx.set_np()

#Plot ReLu function
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize = (5, 2.5))
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize = (5, 2.5))

#Plot Sigmoid function
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize = (5, 2.5))
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize = (5, 2.5))

#Plot tanh function
with autograd.record():
    y = np.tanh(x) #npx doesnt have tanh function
d2l.plot(x, y, 'x', 'tanh(x)', figsize = (5, 2.5))
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize = (5, 2.5))

#Calculate the derivative of the pReLU activation function
#with autograd.record():
#    y = npx.relu(x) + 0.01 * min(0, x)
#d2l.plot(x, y, 'x', 'prelu(x)', figsize = (5, 2.5))
#y.backward()
#d2l.plot(x, x.grad, 'x', 'grad of relu', figsize = (5, 2.5))
