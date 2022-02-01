# necessary packages/libraries/modules
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import torch
import copy
import math
import random
from statistics import median
import os
from scipy.stats import beta
import torch.distributions as tdist
import time
import typing
from random import randint
import pickle as pkl


# distribution functions
w_dist1 = tdist.Normal(0, 1.0)
w_dist2 = tdist.Normal(0, 1.0)


# Instead of generating Wiener noise term at every iteration
# we randomly select one of the generated ones to save time
#
def get_w(iter_i, iter_n, w1, w2):

    # make the following a comment line to use the later part
    # return w2[randint(0, 9)]

    # the following part improves the results, however, it makes it less realistic to simulate CIM
    # I generated 2 random numbers set from different distributions (same mean different std parameters)
    # The one with the larger std is used at the first 40% of the iterations
    # The one with the smaller std is used if the iter number is between 0.4*total_iteration and 0.8*total_iteration
    # In the las 20% of the iterations there is no noise
    if iter_i > 0.4 * iter_n:
        if iter_i > 0.8 * iter_n:
            return w1[randint(0, 19)]
        else:
            return w2[randint(0, 19)]
    else:
        return w1[randint(0, 19)]


# to calculate the B matrix (B from the paper that explaines how to use weighted Motkin-Straus method)
# A matrix is the adjaceny matrix, w is node weights
# device is to keep the tensors compatible
def get_b_matrix(A, w, device):
    B = np.zeros(A.shape)
    bi = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(i, A.shape[0]):
            if A[i, j] == 0:
                B[i, j] = B[j, i] = (
                    5 * ((1 / w[i]) + (1 / w[j])) / 2
                )  # 5 here needs to match at the objective function calculation
            else:
                B[i, j] = B[j, i] = 0

    return torch.Tensor(B).to(device)


# Currently we do not use this function
# Hoever, if you need to see how spins progress then you may use this (it will slow the experiment)
def keep_history(a_hist, a_prev, i_iter, dim):
    for i_spin in range(dim):
        a_hist[i_iter, i_spin] = a_prev[i_spin].item()
    return a_hist


# main function where the simulation happens


def sde_MF_CIM_weighted_vanilla(
    dim, a_matrix, b_matrix, weights, pmax, iter_n, batch_size=1, device="cpu"
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

    # Two different set of random variables are created based on two different dist functions
    w1 = [w_dist1.sample((batch_size, dim, 1)).to(device) for _ in range(20)]
    w2 = [w_dist2.sample((batch_size, dim, 1)).to(device) for _ in range(20)]

    # main loop for the simulation
    for iter_i in range(iter_n):
        # the follwing line is the pumping schedule
        if pmax > -1:
            p0 = p_max * (iter_i) / (iter_n)
        else:
            p0 = p_max

        # getting the noise
        w = get_w(iter_i, iter_n, w1, w2)
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

        if False:
            term2 = -torch.einsum("cj,ij->ci", mu_tilde.pow(2), b_matrix) * mu_tilde
            term2_cst = 0.08 * j / np.sqrt(float(torch.sum(torch.abs(b_matrix))) / dim)
            # term2 *= term2_cst
            # print(term2_cst)
            term2 *= 0.0008
        else:
            term2 = 0.00 #Farhad said in Slack message to set term2 to zero

        term3 = np.sqrt(j) * (sigma - 0.5) * Wt

        mu += dt * (term1 + term2 + term3)

        term1_sg = 2 * (-(1 + j) + p - 3 * g ** 2 * mu.pow(2)) * sigma
        term2_sg = -2 * j * (sigma - 0.5).pow(2)
        term3_sg = (1 + j) + 2 * g ** 2 * mu.pow(2)

        sigma += dt * (term1_sg + term2_sg + term3_sg)

        # this part is not necessasy but to keep thing under contraol just in case there is something wrong
        # c = torch.clamp(c, min=-10, max=10)

    # computing the engery of the isin model
    clique_obj, clique_loc = compute_energy_GSP(
        mu_tilde, a_matrix, weights, batch_size, device
    )
    return mu_tilde, clique_obj


# this function is called from another files to initialze the cvalue
def solve_weighted_vanilla(batch_size, adjancy, weights, size, pmax, iters):
    """
    batch_size: how many times we solve the same problem at once. Default value is 1
    adjancy: Adjancy matrix
    weights: node weights for the weighted maximum clique problem
    size: number of nodes in the max clique problem
    pmax: max value for the pumping parameter
    """
    # if the results are getting worse for the larger problems then increasing the iter_n may help
    iter_n = iters

    device = "cpu"

    # the following line chekcs if there is a GPU. If so it uses otherwise CPU is used
    if torch.cuda.is_available():
        device = "cuda"
    b_size = batch_size
    b_matrix = get_b_matrix(adjancy, weights, device)
    dim = size
    # to keep the elapsed time we store the starting time
    start_time = time.time()
    # dim: node size of the graph
    # pmax: it is for the pump scheduling
    c_vals, clique_obj_val = sde_MF_CIM_weighted_vanilla(
        dim, adjancy, b_matrix, weights, pmax, iter_n, batch_size=b_size, device=device
    )
    elapsed_time = round(time.time() - start_time, 4)  # elapsed time is calculated

    return c_vals, clique_obj_val, elapsed_time
