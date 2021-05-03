#Reading the Dataset

#%matplotpib inline
import random
from mxnet import autograd, np, npx
from d2l import mxnet as d2l
npx.set_np()

def data_iter(batch_size, features, labels): #@save
    num_examples = len(features)
    indices = list(range(num_examples))
    #The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

batch_size = 100
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
