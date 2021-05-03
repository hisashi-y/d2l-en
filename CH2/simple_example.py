from mxnet import autograd, np, npx
from IPython import display
from d2l import mxnet as d2l

npx.set_np()

x = np.arange(4.0)
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
with autograd.record():
    y = 2 * np.dot(x, x)
y.backward()
x.grad
x.grad == 4 * x
with autograd.record():
    y = x.sum()

y.backward()
x.grad

# When we invoke `backward` on a vector-valued variable `y` (function of `x`),
# a new scalar variable is created by summing the elements in `y`. Then the
# gradient of that scalar variable with respect to `x` is computed
with autograd.record():
    y = x * x # 'y' is a vector
y.backward()
x.grad

with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
y.backward()
x.grad == 2 * x

def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
a.grad

def f(p):
    return np.sin(p)

p = np.arange(0, 100, 0.01)
p.attach_grad()
with autograd.record():
    r = f(p)
r.backward()
p.grad

d2l.plot(p, [np.sin(p), 2 * p - 3], 'p', 'f(p)', legend = ['f(p)', 'Tangent line (p = 1)'] )
