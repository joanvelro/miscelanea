import numpy as np
import time
np.random.seed(0)

I = np.arange(0, 100)
J = np.arange(0, 100)
K = np.arange(0, 100)
S = np.arange(0, 30)
print('Building variable as dict ...')
A = {x: [] for x in [tuple(I), tuple(J), tuple(K)]}

T = []
print('Nested three-loop ...')
for s in S:
    start = time.process_time()
    print('{}/{}'.format(s, len(S)))
    for i in I:
        for j in J:
            for k in K:
                A[(i, j, k)] = np.random.random()
    end = time.process_time()
    t = end - start
    T.append(t)
print('Avg. execution time:{}s'.format(np.mean(T)))
