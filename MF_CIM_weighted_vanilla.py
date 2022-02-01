# necessary packages/libraries/modules
import numpy as np
import torch
import torch.distributions as tdist

def sde_MF_CIM_weighted_vanilla(
    dim,
    #a_matrix, #b_matrix, 
    #weights,
    pmax, iter_n, batch_size=1, device="cpu"
):
    """
    dim: dimension of the graph
    b_matrix: B matrix for weighted Motzkin-Straus model
    pmax: maximum value for the pumping parameter
    iter_n: Total number of iterations to run the simulation
    batch_size: how many times we solve the same problem at once. Default value is 1
    device: which hardware to use to do the simulation
    """
    p_max = pmax
    j = 0.5
    g = 1e-2
    dt = 0.025  # Normalized roundtrip time
    # Tensor for the spins is created at the following line
    mu = torch.zeros((batch_size, dim)).to(device)
    sigma = torch.ones((batch_size, dim)).to(device) * (1 / 4)
    ones = torch.ones([dim]).to(device)

    # distribution functions
    w_dist = tdist.Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
    
    # main loop for the simulation
    for iter_i in range(iter_n):
        # the follwing line is the pumping schedule
        if pmax > -1:
            p0 = p_max * (iter_i) / (iter_n)
        else:
            p0 = p_max

        # getting the noise
        w = w_dist.sample((batch_size, dim, 1))
        Wt = w[
            :, :, 0:1
        ].squeeze()  # since the return value is something like [ [vector as my values]]
        Wt *= np.sqrt(dt)

        mu_tilde = mu + np.sqrt(1 / (4 * j)) * Wt
        p = (
            p0
            - 6.0e-4
            * torch.einsum("c,i -> ci", torch.sum(mu_tilde.pow(2), dim=1), ones)
            + g ** 2 * mu_tilde.pow(2)
        )

        term1 = (-(1 + j) + p - g ** 2 * mu.pow(2)) * mu

        term2 = 0.00 #Farhad said in Slack message to set term2 to zero

        term3 = np.sqrt(j) * (sigma - 0.5) * Wt

        mu += dt * (term1 + term2 + term3)

        term1_sg = 2 * (-(1 + j) + p - 3 * g ** 2 * mu.pow(2)) * sigma
        term2_sg = -2 * j * (sigma - 0.5).pow(2)
        term3_sg = (1 + j) + 2 * g ** 2 * mu.pow(2)

        sigma += dt * (term1_sg + term2_sg + term3_sg)

        # this part is not necessasy but to keep thing under contraol just in case there is something wrong
        # c = torch.clamp(c, min=-10, max=10)

    return mu_tilde
