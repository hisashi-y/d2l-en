from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation = 'relu'))
    net.add(nn.Dense(16, activation = 'relu'))
    return net

def block2():
    net = nn.Sequential()
    for _ in range(4):
        #Nested here
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()

net = nn.Sequential()
shared = nn.Dense(8, activation = 'relu')
net.add(nn.Dense(8, activation = 'relu'), shared, nn.Dense(8, activation = 'relu', params = shared.params), nn.Dense(10))
net.initialize()

X = np.random.uniform(size = (2, 20))
net(X)

print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
print(net[1].weight.data()[0])
