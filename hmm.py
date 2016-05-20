import numpy as np

class Model:
    """
    hidden markov model
    Parameters
    ----------
    P : Numpy array
        Probability transition matrix for the hidden states
    E : Numpy array
        Emission matrix
    """

    def __init__(self, P, E):
        self.P = P
        self.E = E
        self.pi = self.stationary()

    def stationary(self):
        """
        Calculate the stationary distrobution of the markov chain.
        """
        eig = np.linalg.eig(np.transpose(self.P))
        i = np.nonzero(eig[0] == 1)[0][0] #find eigenvalue of 1.
        e = eig[1][:,i] #select eigenvector with eigenvalue 1.
        return e/sum(e) #scale so that the vector sums to 1.

    def viterbi(self, obs):
        """
        Perform viterbi algoirthm and return the most probable sequence
        of hidden states.
        obs : list or Numpy array of observations
        """
        states = np.arange(len(self.P))
        V = np.zeros((len(states), len(obs)))
        Pointer = np.zeros((len(states), len(obs)), dtype=np.uint64)
        Pointer[:, 0] = np.array([np.nan, np.nan])
        #initialise V using stationary distro of P
        V[:, 0] = self.pi * self.E[:, obs[0]]
        for i, o in enumerate(obs[1:], start=1):
            for m in states:
                l = V[:, i-1] * self.P[:, m]
                max_index =  np.argmax(l)
                V[m, i] = self.E[m, o] * l[max_index]
                Pointer[m, i] = max_index 
        #initialise rho with max probability pointer
        rho = [np.argmax(V[:, -1])] 
        #traceback
        for p in Pointer[:, ::-1].T[:-1, :]: 
            rho.append(p[rho[-1]])
        #reverse rho and return
        return rho[::-1] 

    def forward_backward(self, obs):
        """
        Perform forward backward algoirthm and return the a numpy array 
        of posterior probabilities.
        obs : list or Numpy array of observations
        """
        states = np.arange(len(self.P))
        F = np.zeros((len(states), len(obs)))
        B = np.zeros((len(states), len(obs)))
        #initialise F using stationary disto of P
        F[:, 0] = self.pi * self.E[:, obs[0]] 
        #iterate forward through the observations to calculate F
        for i, o in enumerate(obs[1:], start=1):  
            for m in states:
                F[m, i] = self.E[m, o] * sum(F[:, i-1] * self.P[:, m])
        
        B[:,-1] = np.ones(len(states)) #initialise B as 1's
        #iterate back through the observations to calculate B
        for i, o in enumerate(obs[1:][::-1], start=1): 
            for m in states:
                B[m, -(i+1)] = sum(self.E[:, o] * B[:, -i] * self.P[m, :])
        
        Pr_total = sum(F[:,-1])
        # OR: Pr_total = sum(B.T[0] * self.E[:, obs(0)] * self.pi)
        Posterior = (F * B)/Pr_total
        return Posterior 

