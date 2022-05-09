import math
import numpy as np



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sample_gumbel(eps=1e-20):
    u = np.random.uniform(0,1)
    return -math.log(-math.log(u+eps)+eps)


def gumbel_softmax(logits, tau=0.01):
    g = [sample_gumbel() for _ in range(logits.shape[0])]
    y = logits + g
    return softmax(y/tau)

x = np.array([0 ,1, 2, 3])
p = np.array([4., 3., 5., 7.])

softmax(p)
# array([1.22457910e-04, 9.04848363e-04, 6.68597532e-03, 9.92286718e-01])
sum_ = np.zeros_like(p)
for i in range(10000):
    sum_ += gumbel_softmax(p)
sum_/np.sum(sum_)
# array([9.99887028e-05, 9.97102104e-04, 7.85516093e-03, 9.91047748e-01])

import matplotlib.pyplot as plt

plt.plot(x,softmax(p))
plt.show()
plt.plot(x,sum_/np.sum(sum_))


