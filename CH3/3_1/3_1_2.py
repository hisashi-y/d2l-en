import math
import time
from mxnet import np
from d2l import mxnet as d2l

n = 10000
a = np.ones(n)
b = np.ones(n)

class Timer: #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""

    def cumsum(self):
        """Return the accumulated time."""
        #np.arrayでnparrayを生成、cumsumでその和を出す。tolistでpython標準のリスト型に戻す
        return np.array(self.times).cumsum().tolist()

c = np.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f}sec'

timer.start()
d = a + b
f'{timer.stop():.5f}sec'
