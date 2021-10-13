import numpy as np
import itertools
import time

np.random.seed(0)

I = np.arange(0, 100)
J = np.arange(0, 100)
K = np.arange(0, 100)
S = np.arange(0, 30)
print('Building variable as dict ...')
A = {x: [] for x in [tuple(I), tuple(J), tuple(K)]}


T = []
print('itertools loop ...')
for s in S:
    print('{}/{}'.format(s, len(S)))
    start = time.process_time()
    for i, j, k in itertools.product(I, J, K):
        A[(i, j, k)] = np.random.random()
    end = time.process_time()
    t = end - start
    T.append(t)
print('Avg. execution time:{}s'.format(np.mean(T)))
