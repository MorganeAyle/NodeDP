import math
import pdb

import numpy as np

from autodp.mechanism_zoo import ExactGaussianMechanism
from autodp.transformer_zoo import Composition, AmplificationBySampling

from src.utils import compute_hypergeometric
from src.constants import SENSITIVITY_ONE, POISSON_ACCOUNTANT, TRANSDUCTIVE_DATASETS


def compute_sampling_distribution(method, num_train_nodes, num_roots, depth, sampler_args, num_total_nodes, dataset):
    if method == 'drw':
        if dataset in TRANSDUCTIVE_DATASETS:
            denom_nodes = num_total_nodes
        else:
            denom_nodes = num_train_nodes
        # probability of sampling a node as the root
        p_root = num_roots / num_train_nodes
        # probability of sampling a node in a random walk
        p_rw = (sampler_args["max_degree"]) / (denom_nodes - num_roots - (num_roots - 1) * depth)
        for i in reversed(range(num_roots - 1)):
            prev_p_rw = (sampler_args["max_degree"]) / (denom_nodes - num_roots - i * depth)
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

    elif method == 'uniform':
        rho_1 = num_roots / num_train_nodes
        rho = [1-rho_1, rho_1]

    else:
        raise NotImplementedError(f"Unknown method {method}.")

    return rho


class PrivacyAccountant:
    def __init__(self, training_args, sampler_args, depth, num_train_nodes, clip_norm, fout, num_total_nodes, dataset):
        self.train_args = training_args
        self.method = sampler_args["method"]
        self.accountant_name = training_args["accountant"]
        self.clip_norm = clip_norm
        self.num_roots = sampler_args["num_root"]
        self.depth = depth
        self.noise_multiplier = self.train_args['noise_multiplier']

        self.gamma = 0
        self.epsilon = 0
        self.delta = 0
        self.steps = 0

        max_sampled_nodes = 1 if self.method in SENSITIVITY_ONE else (sampler_args["max_degree"] ** (self.depth + 1)
                                                                      - 1) // (sampler_args["max_degree"] - 1)
        cst = 1 if self.accountant_name in POISSON_ACCOUNTANT else 2
        self.sensitivity = cst * max_sampled_nodes * clip_norm
        self.sigma = self.noise_multiplier * self.sensitivity
        self.distribution = compute_sampling_distribution(self.method, num_train_nodes, self.num_roots, self.depth,
                                                          sampler_args, num_total_nodes, dataset)

        fout(f"Max sampled nodes: {max_sampled_nodes}, Sensitivity: {self.sensitivity}, Sigma: {self.sigma}, "
             f"Noise multiplier: {self.noise_multiplier}, Gradients norm: {self.clip_norm}, "
             f"Prob. of sampling v: {sum(self.distribution[1:])}")

    def one_step(self):
        """
        Keeps track of the privacy spent at every additional step.
        """
        self.steps += 1

        if self.accountant_name == 'rdp_poisson_autodp':
            self.delta = self.train_args['delta']
            gm1 = ExactGaussianMechanism(self.noise_multiplier, name='GM1')
            prob = self.distribution[1]
            compose = Composition()
            poisson_sample = AmplificationBySampling(PoissonSampling=True)
            composed_poisson_mech = compose([poisson_sample(gm1, prob)], [self.steps])

            self.delta = self.train_args['delta']
            self.epsilon = composed_poisson_mech.get_approxDP(self.delta)

        elif self.accountant_name == 'rdp_uniform_autodp':
            self.delta = self.train_args['delta']
            gm1 = ExactGaussianMechanism(self.noise_multiplier, name='GM1')
            gm1.replace_one = True
            prob = self.distribution[1]
            compose = Composition()
            uniform_sample = AmplificationBySampling(PoissonSampling=False)
            composed_uniform_mech = compose([uniform_sample(gm1, prob)], [self.steps])

            self.delta = self.train_args['delta']
            self.epsilon = composed_uniform_mech.get_approxDP(self.delta)

        elif self.accountant_name == "baseline":
            # Compute RDP parameters
            alpha = self.train_args["alpha"]
            self.gamma += 1 / (alpha - 1) * np.log(sum(np.array([p * (
                np.exp(alpha * (alpha - 1) * (i * 2 * self.clip_norm) ** 2 / (
                            2 * self.sigma ** 2))) for i, p in enumerate(self.distribution)])))

            # Convert to DP parameters
            self.epsilon = self.gamma + np.log(1 / self.train_args['delta']) / (alpha - 1)
            self.delta = self.train_args['delta']

        else:
            raise NotImplementedError

    def log(self, fout):
        fout("DP: (" + str(self.epsilon) + "," + str(self.delta) + ")")
