# necessary packages/libraries/modules
import numpy as np
import torch
import torch.distributions as tdist

class CIMSDE():
    def __init__(self, dim, batch_size=1, device='cpu'):
        self.dim = dim 
        self.batch_size = batch_size
        self.device = device
    
        self.j = 0.5 
        self.g = 1e-2
        self.dt = 0.025  # Normalized roundtrip time
        
        self.mu = torch.zeros((batch_size, dim)).to(device)
        self.sigma = torch.ones((batch_size, dim)).to(device) * (1 / 4)
        self.ones = torch.ones([dim]).to(device)

        self.w_dist = tdist.Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))

        self.initialize() 

    def initialize(self):
        self.iter_i = 0

    # main loop for the simulation
    def iteration(self, pump=0.5):

        # getting the noise
        w = self.w_dist.sample((self.batch_size, self.dim, 1))
        Wt = w[
            :, :, 0:1
        ].squeeze()  # since the return value is something like [ [vector as my values]]
        Wt *= np.sqrt(self.dt)

        mu_tilde = self.mu + np.sqrt(1 / (4 * self.j)) * Wt
        p = (
            pump
            - 6.0e-4
            * torch.einsum("c,i -> ci", torch.sum(mu_tilde.pow(2), dim=1), self.ones)
            + self.g ** 2 * mu_tilde.pow(2)
        )

        term1 = (-(1 + self.j) + p - self.g ** 2 * self.mu.pow(2)) * self.mu

        term2 = 0.00 #Farhad said in Slack message to set term2 to zero

        term3 = np.sqrt(self.j) * (self.sigma - 0.5) * Wt

        self.mu += self.dt * (term1 + term2 + term3)

        term1_sg = 2 * (-(1 + self.j) + p - 3 * self.g ** 2 * self.mu.pow(2)) * self.sigma
        term2_sg = -2 * self.j * (self.sigma - 0.5).pow(2)
        term3_sg = (1 + self.j) + 2 * self.g ** 2 * self.mu.pow(2)

        self.sigma += self.dt * (term1_sg + term2_sg + term3_sg)

        # this part is not necessasy but to keep thing under contraol just in case there is something wrong
        # c = torch.clamp(c, min=-10, max=10)

        return mu_tilde
