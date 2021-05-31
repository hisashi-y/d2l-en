from mxnet import np, npx
from mxnet.gluon import nn

npx.set_np()

#net = nn.Sequential()
#net.add(nn.Dense(256, activation='relu'))
#net.add(nn.Dense(10))
#net.initialize()

X = np.random.uniform(size=(2, 20))
#net(X)

class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two
    # fully-connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.out = nn.Dense(10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        return self.out(self.hidden(X))

#net = MLP()
#net.initialize()
#net(X)

class MySequential(nn.Block):
    def add(self, block):
        self._children[block.name] = block

    def forward(self, X):
        for block in self._children.values():
            X = block(X)
        return X

net = MySequential()
net.add(nn.Dense(256, activation = 'relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)

class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rand_weight = self.params.get_constant('rand_weight', np.random.uniform(size = (20, 20)))
        self.dense = nn.Dense(20, activation = 'relu')

    def forward(self, X):
        X = self.dense(X)
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        X = self.dense(X)
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
