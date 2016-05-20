import hmm
import numpy as np

P = np.array([[.5,.5], [.2,.8]])
E = np.array([[.7, .3], [.3, .7]])
observed = [1, 0, 1, 0, 0, 0, 1, 0]

model = hmm.Model(P,E)
print model.viterbi(observed)
print model.forward_backward(observed)

