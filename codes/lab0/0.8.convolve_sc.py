import numpy as np
from scipy import signal

x = np.array([1,2,3,4])
y = signal.convolve(x, x)
print(x,y)

