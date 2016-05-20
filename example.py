import hmm
import numpy as np

P = np.array([[.5,.5], [.2,.8]])
E = np.array([[.7, .3], [.3, .7]])
observed = [1, 0, 1, 0, 0, 0, 1, 0]

model = hmm.Model(2,P,E)
print model.viterbi(observed)
print model.forward_backward(observed)

#from http://homepages.ulb.ac.be/~dgonze/TEACHING/viterbi.pdf

S = 'GGCACTGAA'
encode = {'A':0, 'C':1, 'G':2, 'T':3}
observed = []
for s in S:
    observed.append(encode[s])

num_states = 2 #H: 0, L: 1
P = np.array([[.5, .5],[.4, .6]])
E = np.array([[.2, .3, .3, .2], [.3, .2, .2, .3]])
model = hmm.Model(2, P, E, initial_distro = [.5, .5])
print model.viterbi(observed)
print model.forward(observed)


