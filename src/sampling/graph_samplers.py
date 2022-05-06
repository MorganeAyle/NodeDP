import numpy as np
import cython_sampler as cy


class GraphSampler:
    def __init__(self, adj_train, node_train):
        self.adj_train = adj_train
        self.node_train = np.unique(node_train).astype(np.int32)

    def par_sample(self):
        ret = self.cy_sampler.par_sample()
        return ret


class RandomWalks(GraphSampler):
    """
    The sampler performs unbiased random walk, by following the steps:
     1. Randomly pick `size_root` number of root nodes from all training nodes;
     2. Perform length `size_depth` random walk from the roots. The current node
            expands the next hop by selecting one of the neighbors uniformly
            at random;
     3. Generate node-induced subgraph from the nodes touched by the random walk.
    """
    def __init__(self, adj_train, node_train, size_root, size_depth, num_par_sampler, samples_per_proc):
        self.size_root = size_root
        self.size_depth = size_depth
        super().__init__(adj_train, node_train)
        self.cy_sampler = cy.RW(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            num_par_sampler,
            samples_per_proc,
            self.size_root,
            self.size_depth
        )


class DisjointRandomWalks(GraphSampler):
    def __init__(self, adj_train, node_train, size_root, size_depth, num_par_sampler, samples_per_proc):
        self.size_root = size_root
        self.size_depth = size_depth
        super().__init__(adj_train, node_train)
        self.cy_sampler = cy.DisjointRW(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            num_par_sampler,
            samples_per_proc,
            self.size_root,
            self.size_depth
        )


class NodesUniformMaxDegree(GraphSampler):
    def __init__(self, adj_train, node_train, size_nodes, max_degree, num_par_sampler, samples_per_proc):
        self.size_nodes = size_nodes
        self.max_degree = max_degree
        super().__init__(adj_train, node_train)
        self.cy_sampler = cy.NodesUniformMaxDegree(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            num_par_sampler,
            samples_per_proc,
            self.size_nodes,
            self.max_degree
        )


class FullBatch(GraphSampler):
    """
    Strictly speaking, this is not a sampler. It simply returns the full adj
    matrix of the training graph. This can serve as a baseline to compare
    full-batch vs. minibatch performance.

    Therefore, the size_subgraph argument is not used here.
    """
    def __init__(self, adj_train, node_train, num_par_sampler, samples_per_proc):
        super().__init__(adj_train, node_train)
        self.cy_sampler = cy.FullBatch(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            num_par_sampler,
            samples_per_proc
        )
