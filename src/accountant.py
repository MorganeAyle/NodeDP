import math
import pdb

import numpy as np

from src.utils import compute_hypergeometric
from src.constants import SENSITIVITY_ONE, RDP_ACCOUNTANT


def compute_sampling_distribution(method, num_train_nodes, num_roots, depth, sampler_args):
    if method == 'drw':
        # probability of sampling a node as the root
        p_root = num_roots / num_train_nodes
        # probability of sampling a node in a random walk
        p_rw = (sampler_args["max_degree"] + depth - 1) / (num_train_nodes - num_roots - (num_roots - 1) * depth)
        for i in reversed(range(num_roots - 1)):
            prev_p_rw = (sampler_args["max_degree"] + depth - 1) / (num_train_nodes - num_roots - i * depth)
            p_rw = prev_p_rw + (1 - prev_p_rw) * p_rw
        p_sample = p_root + (1 - p_root) * p_rw
        rho = [1 - p_sample, p_sample]

    elif method == 'pre_drw':
        rho = compute_hypergeometric(math.ceil(num_train_nodes / (depth + 1)), 1, num_roots)

    elif method == 'pre_drw_w_restarts':
        rho = compute_hypergeometric(math.ceil(num_train_nodes / (depth * sampler_args['restarts'] + 1)), 1, num_roots)

    elif method == 'baseline':
        max_sampled_nodes = (sampler_args["max_degree"] ** (depth + 1) - 1) // (sampler_args["max_degree"] - 1)
        rho = compute_hypergeometric(num_train_nodes, max_sampled_nodes, num_roots)

    else:
        raise NotImplementedError(f"Unknown method {method}.")

    return rho


class PrivacyAccountant:
    def __init__(self, training_args, sampler_args, depth, num_train_nodes, clip_norm, fout):
        self.train_args = training_args
        self.method = sampler_args["method"]
        self.accountant = training_args["accountant"]
        self.clip_norm = clip_norm
        self.num_roots = sampler_args["num_root"]
        self.depth = depth

        self.gamma = 0
        self.epsilon = 0
        self.delta = 0

        max_sampled_nodes = 1 if self.method in SENSITIVITY_ONE else (sampler_args["max_degree"] ** (self.depth + 1)
                                                                      - 1) // (sampler_args["max_degree"] - 1)
        self.sensitivity = 2 * max_sampled_nodes * clip_norm
        sigma_without_norm = 2 * max_sampled_nodes
        self.sigma = sigma_without_norm * clip_norm if not training_args['sigma'] else training_args['sigma']
        self.distribution = compute_sampling_distribution(self.method, num_train_nodes, self.num_roots, self.depth,
                                                          sampler_args)

        fout(f"Max sampled nodes: {max_sampled_nodes}, Sensitivity: {self.sensitivity}, Sigma: {self.sigma}, "
             f"Gradients norm: {self.clip_norm}, Prob. of not sampling v: {self.distribution[0]}")

    def one_step(self):
        if self.accountant in RDP_ACCOUNTANT:
            # Compute RDP parameters
            alpha = self.train_args["alpha"]
            if self.accountant == "baseline":
                self.gamma += 1 / (alpha - 1) * np.log(sum(np.array([p * (
                    np.exp(alpha * (alpha - 1) * (i * 2 * self.clip_norm) ** 2 / (
                            2 * self.sigma ** 2))) for i, p in enumerate(self.distribution)])))

            elif self.accountant == "sub_rdp":
                p = self.distribution[1]
                rdp_unamp = lambda x: (x * self.sensitivity ** 2) / (2 * self.sigma ** 2)
                self.gamma += 1 / (alpha - 1) * np.log(
                    1 + p ** 2 * math.comb(alpha, 2) * min(4 * (np.exp(rdp_unamp(2)) - 1), 2 * np.exp(rdp_unamp(2))))

            # Convert to DP parameters
            self.epsilon = self.gamma + np.log(1 / self.train_args['delta']) / (alpha - 1)
            self.delta = self.train_args['delta']

        elif self.accountant == "std":
            # Compute DP parameters
            eps = self.train_args["eps"]
            self.epsilon += self.distribution[1] * (np.exp(eps) - 1)
            self.delta += self.distribution[1] * self.train_args["delta"]

    def log(self, fout):
        if self.accountant in RDP_ACCOUNTANT:
            fout("RDP: (" + str(self.train_args['alpha']) + "," + str(self.gamma) + ")")

        fout("DP: (" + str(self.epsilon) + "," + str(self.delta) + ")")
