from mxnet import np, npx
npx.set_np()
from mxnet import autograd
from d2l import mxnet as d2l
def f(x):
      return np.sin(x)

x = np.linspace(- np.pi,np.pi,100)
x.attach_grad()
with autograd.record():
        y = f(x)

y.backward()
d2l.plot(x,(y,x.grad),legend = [('sin(x)','cos(x)')])

scores = {'network' : 88, 'database' : 95, 'security' : 90, 'software' : 100}
total = sum(scores.values())
avg = total / len(scores.items())

print('合計点:{}'.format(total))
print('平均点:{}'.format(avg))
