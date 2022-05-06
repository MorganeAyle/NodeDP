# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args = -fopenmp -std=c++11
# distutils: extra_link_args = -fopenmp
cimport cython
from cython.parallel import prange,parallel
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort, unique, lower_bound
from libc.stdio cimport printf
cimport numpy as np
import numpy as np
from libcpp cimport bool
from libc.stdlib cimport rand
cdef extern from "stdlib.h":
    int RAND_MAX

cimport cython_utils as cutils
import cython_utils as cutils


cdef class BaseSampler:
    cdef int num_proc, num_sample_per_proc
    cdef vector[int] adj_indptr_vec
    cdef vector[int] adj_indices_vec
    cdef vector[int] node_train_vec
    cdef vector[vector[int]] node_sampled
    cdef vector[vector[int]] ret_indptr
    cdef vector[vector[int]] ret_indices
    cdef vector[vector[int]] ret_indices_orig
    cdef vector[vector[float]] ret_data
    cdef vector[vector[int]] ret_edge_index

    def __cinit__(self, np.ndarray[int, ndim=1, mode='c'] adj_indptr,
                  np.ndarray[int, ndim=1, mode='c'] adj_indices,
                  np.ndarray[int, ndim=1, mode='c'] node_train,
                  int num_proc, int num_sample_per_proc, *argv):
        cutils.npy2vec_int(adj_indptr, self.adj_indptr_vec)
        cutils.npy2vec_int(adj_indices, self.adj_indices_vec)
        cutils.npy2vec_int(node_train, self.node_train_vec)
        self.num_proc = num_proc
        self.num_sample_per_proc = num_sample_per_proc
        self.node_sampled = vector[vector[int]](num_proc * num_sample_per_proc)
        self.ret_indptr = vector[vector[int]](num_proc * num_sample_per_proc)
        self.ret_indices = vector[vector[int]](num_proc * num_sample_per_proc)
        self.ret_indices_orig = vector[vector[int]](num_proc * num_sample_per_proc)
        self.ret_data = vector[vector[float]](num_proc * num_sample_per_proc)
        self.ret_edge_index = vector[vector[int]](num_proc * num_sample_per_proc)

    cdef void sample(self, int p) nogil:
        pass


cdef class Sampler(BaseSampler):
    def __cinit__(self, np.ndarray[int, ndim=1, mode='c'] adj_indptr,
                  np.ndarray[int, ndim=1, mode='c'] adj_indices,
                  np.ndarray[int, ndim=1, mode='c'] node_train,
                  int num_proc, int num_sample_per_proc, *argv):
        pass

    cdef void adj_extract(self, int p) nogil:
        """
        Extract a subg adj matrix from the original training adj matrix
        ret_indices_orig:   the indices vector corresponding to node id in original G.
        """
        cdef int r = 0
        cdef int idx_g = 0
        cdef int i, i_end, v, j
        cdef int num_v_orig, num_v_sub
        cdef int start_neigh, end_neigh
        cdef vector[int] _arr_bit
        cdef int cumsum
        num_v_orig = self.adj_indptr_vec.size() - 1  # number of total training nodes
        while r < self.num_sample_per_proc:
            _arr_bit = vector[int](num_v_orig, -1)  # create vector of size num_v_orig with all values -1
            idx_g = p * self.num_sample_per_proc + r  # index of subgraph
            num_v_sub = self.node_sampled[idx_g].size()  # number of nodes sampled in this subgraph
            self.ret_indptr[idx_g] = vector[int](num_v_sub + 1, 0)
            self.ret_indices[idx_g] = vector[int]()
            self.ret_indices_orig[idx_g] = vector[int]()
            self.ret_data[idx_g] = vector[float]()
            self.ret_edge_index[idx_g] = vector[int]()
            i_end = num_v_sub
            i = 0
            while i < i_end:
                _arr_bit[self.node_sampled[idx_g][i]] = i  # _arr_bit is -1 everywhere except at the position of
                # the nodes sampled in the subgraph, where it contains the index of the node in self.node_sampled
                i = i + 1
            i = 0
            while i < i_end:
                v = self.node_sampled[idx_g][i]  # iterate over the nodes sampled in subgraph
                start_neigh = self.adj_indptr_vec[v]
                end_neigh = self.adj_indptr_vec[v + 1]
                j = start_neigh
                while j < end_neigh:  # iterate over ALL neighbors of the node
                    if _arr_bit[self.adj_indices_vec[j]] > -1:  # if the neighbor was also sampled
                        self.ret_indices[idx_g].push_back(_arr_bit[self.adj_indices_vec[j]])  # add to self.ret_indices
                        # the index of the neighbor in self.node_sampled (will be the column)
                        self.ret_indices_orig[idx_g].push_back(self.adj_indices_vec[j])  # add to self.ret_indices_org
                        # the column of the neighbor in the original adjacency
                        self.ret_edge_index[idx_g].push_back(j)  # add to ret_edge_index the index of the neighbor in
                        # self.adj_indices_vec
                        self.ret_indptr[idx_g][_arr_bit[v] + 1] = self.ret_indptr[idx_g][_arr_bit[v] + 1] + 1
                        self.ret_data[idx_g].push_back(1.)
                    j = j + 1
                i = i + 1
            cumsum = self.ret_indptr[idx_g][0]
            i = 0
            while i < i_end:
                cumsum = cumsum + self.ret_indptr[idx_g][i + 1]
                self.ret_indptr[idx_g][i + 1] = cumsum
                i = i + 1
            r = r + 1

    def get_return(self):
        """
        Convert the subgraph related data structures from C++ to python. So that cython
        can return them to the PyTorch trainer.

        Inputs:
            None

        Outputs:
            see outputs of the `par_sample()` function.
        """
        num_subg = self.num_proc*self.num_sample_per_proc
        l_subg_indptr = []
        l_subg_indices = []
        l_subg_data = []
        l_subg_nodes = []
        l_subg_edge_index = []
        offset_nodes = [0]
        offset_indptr = [0]
        offset_indices = [0]
        offset_data = [0]
        offset_edge_index = [0]
        for r in range(num_subg):
            offset_nodes.append(offset_nodes[r]+self.node_sampled[r].size())
            offset_indptr.append(offset_indptr[r]+self.ret_indptr[r].size())
            offset_indices.append(offset_indices[r]+self.ret_indices[r].size())
            offset_data.append(offset_data[r]+self.ret_data[r].size())
            offset_edge_index.append(offset_edge_index[r]+self.ret_edge_index[r].size())
        cdef vector[int] ret_nodes_vec = vector[int]()
        cdef vector[int] ret_indptr_vec = vector[int]()
        cdef vector[int] ret_indices_vec = vector[int]()
        cdef vector[int] ret_edge_index_vec = vector[int]()
        cdef vector[float] ret_data_vec = vector[float]()
        ret_nodes_vec.reserve(offset_nodes[num_subg])
        ret_indptr_vec.reserve(offset_indptr[num_subg])
        ret_indices_vec.reserve(offset_indices[num_subg])
        ret_data_vec.reserve(offset_data[num_subg])
        ret_edge_index_vec.reserve(offset_edge_index[num_subg])
        for r in range(num_subg):
            # if type(self) is RW_NO:
            #     ret_nodes_vec.insert(ret_nodes_vec.end(), self.root_sampled[r].begin(), self.root_sampled[r].end())
            ret_nodes_vec.insert(ret_nodes_vec.end(),self.node_sampled[r].begin(),self.node_sampled[r].end())
            ret_indptr_vec.insert(ret_indptr_vec.end(),self.ret_indptr[r].begin(),self.ret_indptr[r].end())
            ret_indices_vec.insert(ret_indices_vec.end(),self.ret_indices[r].begin(),self.ret_indices[r].end())
            ret_edge_index_vec.insert(ret_edge_index_vec.end(),self.ret_edge_index[r].begin(),self.ret_edge_index[r].end())
            ret_data_vec.insert(ret_data_vec.end(),self.ret_data[r].begin(),self.ret_data[r].end())

        cdef cutils.array_wrapper_int wint_indptr = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_int wint_indices = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_int wint_nodes = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_float wfloat_data = cutils.array_wrapper_float()
        cdef cutils.array_wrapper_int wint_edge_index = cutils.array_wrapper_int()

        wint_indptr.set_data(ret_indptr_vec)
        ret_indptr_np = np.frombuffer(wint_indptr,dtype=np.int32)
        wint_indices.set_data(ret_indices_vec)
        ret_indices_np = np.frombuffer(wint_indices,dtype=np.int32)
        wint_nodes.set_data(ret_nodes_vec)
        ret_nodes_np = np.frombuffer(wint_nodes,dtype=np.int32)
        wfloat_data.set_data(ret_data_vec)
        ret_data_np = np.frombuffer(wfloat_data,dtype=np.float32)
        wint_edge_index.set_data(ret_edge_index_vec)
        ret_edge_index_np = np.frombuffer(wint_edge_index,dtype=np.int32)

        for r in range(num_subg):
            l_subg_nodes.append(ret_nodes_np[offset_nodes[r]:offset_nodes[r+1]])
            l_subg_indptr.append(ret_indptr_np[offset_indptr[r]:offset_indptr[r+1]])
            l_subg_indices.append(ret_indices_np[offset_indices[r]:offset_indices[r+1]])
            l_subg_data.append(ret_data_np[offset_data[r]:offset_data[r+1]])
            l_subg_edge_index.append(ret_edge_index_np[offset_indices[r]:offset_indices[r+1]])

        return l_subg_indptr,l_subg_indices,l_subg_data,l_subg_nodes,l_subg_edge_index

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def par_sample(self):
        """
        The main function for the sampler class. It launches multiple independent samplers
        in parallel (task parallelism by openmp), where the serial sampling function is defined
        in the corresponding sub-class. Then it returns node-induced subgraph by `_adj_extract()`,
        and convert C++ vectors to python lists / numpy arrays by `_get_return()`.

        Suppose we sample P subgraphs in parallel. Each subgraph has n nodes and e edges.

        Inputs:
            None

        Outputs (elements in the list of `ret`):
            l_subg_indptr       list of np array, length of list = P and length of each array is n+1
            l_subg_indices      list of np array, length of list = P and length of each array is m.
                                node IDs in the array are renamed to be subgraph ID (range: 0 ~ n-1)
            l_subg_data         list of np array, length of list = P and length of each array is m.
                                Normally, values in the array should be all 1.
            l_subg_nodes        list of np array, length of list = P and length of each array is n.
                                Element i in the array shows the training graph node ID of the i-th
                                subgraph node.
            l_subg_edge_index   list of np array, length of list = P and length of each array is m.
                                Element i in the array shows the training graph edge index of the
                                i-the subgraph edge.
        """
        cdef int p = 0
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc,schedule='dynamic'):
                self.sample(p)
                self.adj_extract(p)
        ret = self.get_return()
        _len = self.num_proc*self.num_sample_per_proc
        self.node_sampled.swap(vector[vector[int]](_len))
        self.ret_indptr.swap(vector[vector[int]](_len))
        self.ret_indices.swap(vector[vector[int]](_len))
        self.ret_indices_orig.swap(vector[vector[int]](_len))
        self.ret_data.swap(vector[vector[float]](_len))
        self.ret_edge_index.swap(vector[vector[int]](_len))
        return ret


cdef class MaxDegreeSampler(BaseSampler):
    cdef vector[vector[vector[int]]] neighbors
    def __cinit__(self, np.ndarray[int, ndim=1, mode='c'] adj_indptr,
                  np.ndarray[int, ndim=1, mode='c'] adj_indices,
                  np.ndarray[int, ndim=1, mode='c'] node_train,
                  int num_proc, int num_sample_per_proc, *argv):
        cdef int num_v_orig
        num_v_orig = self.adj_indptr_vec.size() - 1
        self.neighbors = vector[vector[vector[int]]](num_proc * num_sample_per_proc, vector[vector[int]](num_v_orig))

    cdef void adj_extract(self, int p) nogil:
        """
        Extract a subg adj matrix from the original training adj matrix
        ret_indices_orig:   the indices vector corresponding to node id in original G.
        """
        cdef int r = 0
        cdef int idx_g = 0
        cdef int i, i_end, v, j
        cdef int num_v_orig, num_v_sub
        cdef int start_neigh, end_neigh
        cdef vector[int] _arr_bit
        cdef int cumsum
        cdef int k = 0
        cdef bool found = False
        num_v_orig = self.adj_indptr_vec.size() - 1  # number of total training nodes
        while r < self.num_sample_per_proc:
            _arr_bit = vector[int](num_v_orig, -1)  # create vector of size num_v_orig with all values -1
            idx_g = p * self.num_sample_per_proc + r  # index of subgraph
            num_v_sub = self.node_sampled[idx_g].size()  # number of nodes sampled in this subgraph
            self.ret_indptr[idx_g] = vector[int](num_v_sub + 1, 0)
            self.ret_indices[idx_g] = vector[int]()
            self.ret_indices_orig[idx_g] = vector[int]()
            self.ret_data[idx_g] = vector[float]()
            self.ret_edge_index[idx_g] = vector[int]()
            i_end = num_v_sub
            i = 0
            while i < i_end:
                _arr_bit[self.node_sampled[idx_g][i]] = i  # _arr_bit is -1 everywhere except at the position of
                # the nodes sampled in the subgraph, where it contains the index of the node in self.node_sampled
                i = i + 1
            i = 0
            while i < i_end:
                v = self.node_sampled[idx_g][i]  # iterate over the nodes sampled in subgraph
                start_neigh = self.adj_indptr_vec[v]
                end_neigh = self.adj_indptr_vec[v + 1]
                j = start_neigh
                while j < end_neigh:  # iterate over ALL neighbors of the node
                    found = False#
                    for k in self.neighbors[idx_g][v]:
                        if k == self.adj_indices_vec[j]:
                            found = True
                    if _arr_bit[self.adj_indices_vec[j]] > -1 and found:  # if the neighbor was also sampled
                        self.ret_indices[idx_g].push_back(_arr_bit[self.adj_indices_vec[j]])  # add to self.ret_indices
                        # the index of the neighbor in self.node_sampled (will be the column)
                        self.ret_indices_orig[idx_g].push_back(self.adj_indices_vec[j])  # add to self.ret_indices_org
                        # the column of the neighbor in the original adjacency
                        self.ret_edge_index[idx_g].push_back(j)  # add to ret_edge_index the index of the neighbor in
                        # self.adj_indices_vec
                        self.ret_indptr[idx_g][_arr_bit[v] + 1] = self.ret_indptr[idx_g][_arr_bit[v] + 1] + 1
                        self.ret_data[idx_g].push_back(1.)
                    j = j + 1
                i = i + 1
            cumsum = self.ret_indptr[idx_g][0]
            i = 0
            while i < i_end:
                cumsum = cumsum + self.ret_indptr[idx_g][i + 1]
                self.ret_indptr[idx_g][i + 1] = cumsum
                i = i + 1
            r = r + 1

    def get_return(self):
        """
        Convert the subgraph related data structures from C++ to python. So that cython
        can return them to the PyTorch trainer.

        Inputs:
            None

        Outputs:
            see outputs of the `par_sample()` function.
        """
        num_subg = self.num_proc*self.num_sample_per_proc
        l_subg_indptr = []
        l_subg_indices = []
        l_subg_data = []
        l_subg_nodes = []
        l_subg_edge_index = []
        offset_nodes = [0]
        offset_indptr = [0]
        offset_indices = [0]
        offset_data = [0]
        offset_edge_index = [0]
        for r in range(num_subg):
            offset_nodes.append(offset_nodes[r]+self.node_sampled[r].size())
            offset_indptr.append(offset_indptr[r]+self.ret_indptr[r].size())
            offset_indices.append(offset_indices[r]+self.ret_indices[r].size())
            offset_data.append(offset_data[r]+self.ret_data[r].size())
            offset_edge_index.append(offset_edge_index[r]+self.ret_edge_index[r].size())
        cdef vector[int] ret_nodes_vec = vector[int]()
        cdef vector[int] ret_indptr_vec = vector[int]()
        cdef vector[int] ret_indices_vec = vector[int]()
        cdef vector[int] ret_edge_index_vec = vector[int]()
        cdef vector[float] ret_data_vec = vector[float]()
        ret_nodes_vec.reserve(offset_nodes[num_subg])
        ret_indptr_vec.reserve(offset_indptr[num_subg])
        ret_indices_vec.reserve(offset_indices[num_subg])
        ret_data_vec.reserve(offset_data[num_subg])
        ret_edge_index_vec.reserve(offset_edge_index[num_subg])
        for r in range(num_subg):
            # if type(self) is RW_NO:
            #     ret_nodes_vec.insert(ret_nodes_vec.end(), self.root_sampled[r].begin(), self.root_sampled[r].end())
            ret_nodes_vec.insert(ret_nodes_vec.end(),self.node_sampled[r].begin(),self.node_sampled[r].end())
            ret_indptr_vec.insert(ret_indptr_vec.end(),self.ret_indptr[r].begin(),self.ret_indptr[r].end())
            ret_indices_vec.insert(ret_indices_vec.end(),self.ret_indices[r].begin(),self.ret_indices[r].end())
            ret_edge_index_vec.insert(ret_edge_index_vec.end(),self.ret_edge_index[r].begin(),self.ret_edge_index[r].end())
            ret_data_vec.insert(ret_data_vec.end(),self.ret_data[r].begin(),self.ret_data[r].end())

        cdef cutils.array_wrapper_int wint_indptr = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_int wint_indices = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_int wint_nodes = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_float wfloat_data = cutils.array_wrapper_float()
        cdef cutils.array_wrapper_int wint_edge_index = cutils.array_wrapper_int()

        wint_indptr.set_data(ret_indptr_vec)
        ret_indptr_np = np.frombuffer(wint_indptr,dtype=np.int32)
        wint_indices.set_data(ret_indices_vec)
        ret_indices_np = np.frombuffer(wint_indices,dtype=np.int32)
        wint_nodes.set_data(ret_nodes_vec)
        ret_nodes_np = np.frombuffer(wint_nodes,dtype=np.int32)
        wfloat_data.set_data(ret_data_vec)
        ret_data_np = np.frombuffer(wfloat_data,dtype=np.float32)
        wint_edge_index.set_data(ret_edge_index_vec)
        ret_edge_index_np = np.frombuffer(wint_edge_index,dtype=np.int32)

        for r in range(num_subg):
            l_subg_nodes.append(ret_nodes_np[offset_nodes[r]:offset_nodes[r+1]])
            l_subg_indptr.append(ret_indptr_np[offset_indptr[r]:offset_indptr[r+1]])
            l_subg_indices.append(ret_indices_np[offset_indices[r]:offset_indices[r+1]])
            l_subg_data.append(ret_data_np[offset_data[r]:offset_data[r+1]])
            l_subg_edge_index.append(ret_edge_index_np[offset_indices[r]:offset_indices[r+1]])

        return l_subg_indptr,l_subg_indices,l_subg_data,l_subg_nodes,l_subg_edge_index

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def par_sample(self):
        """
        The main function for the sampler class. It launches multiple independent samplers
        in parallel (task parallelism by openmp), where the serial sampling function is defined
        in the corresponding sub-class. Then it returns node-induced subgraph by `_adj_extract()`,
        and convert C++ vectors to python lists / numpy arrays by `_get_return()`.

        Suppose we sample P subgraphs in parallel. Each subgraph has n nodes and e edges.

        Inputs:
            None

        Outputs (elements in the list of `ret`):
            l_subg_indptr       list of np array, length of list = P and length of each array is n+1
            l_subg_indices      list of np array, length of list = P and length of each array is m.
                                node IDs in the array are renamed to be subgraph ID (range: 0 ~ n-1)
            l_subg_data         list of np array, length of list = P and length of each array is m.
                                Normally, values in the array should be all 1.
            l_subg_nodes        list of np array, length of list = P and length of each array is n.
                                Element i in the array shows the training graph node ID of the i-th
                                subgraph node.
            l_subg_edge_index   list of np array, length of list = P and length of each array is m.
                                Element i in the array shows the training graph edge index of the
                                i-the subgraph edge.
        """
        cdef int p = 0
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc,schedule='dynamic'):
                self.sample(p)
                self.adj_extract(p)
        ret = self.get_return()
        _len = self.num_proc*self.num_sample_per_proc
        self.node_sampled.swap(vector[vector[int]](_len))
        self.ret_indptr.swap(vector[vector[int]](_len))
        self.ret_indices.swap(vector[vector[int]](_len))
        self.ret_indices_orig.swap(vector[vector[int]](_len))
        self.ret_data.swap(vector[vector[float]](_len))
        self.ret_edge_index.swap(vector[vector[int]](_len))
        return ret


cdef class SamplerNO(BaseSampler):
    cdef vector[vector[int]] root_sampled
    cdef vector[vector[int]] no_rw
    def __cinit__(self, np.ndarray[int, ndim=1, mode='c'] adj_indptr,
                  np.ndarray[int, ndim=1, mode='c'] adj_indices,
                  np.ndarray[int, ndim=1, mode='c'] node_train,
                  int num_proc, int num_sample_per_proc, int size_root, int size_depth):
        cdef int num_v_orig
        num_v_orig = self.adj_indptr_vec.size() - 1
        self.root_sampled = vector[vector[int]](num_proc * num_sample_per_proc)
        self.no_rw = vector[vector[int]](num_proc * num_sample_per_proc, vector[int](num_v_orig, -1))

    cdef void adj_extract(self, int p) nogil:
        """
        Extract a subg adj matrix from the original training adj matrix
        ret_indices_orig:   the indices vector corresponding to node id in original G.
        """
        cdef int r = 0
        cdef int idx_g = 0
        cdef int i, i_end, v, j
        cdef int num_v_orig, num_v_sub
        cdef int start_neigh, end_neigh
        cdef vector[int] _arr_bit
        cdef int cumsum
        num_v_orig = self.adj_indptr_vec.size() - 1  # number of total training nodes
        while r < self.num_sample_per_proc:
            _arr_bit = vector[int](num_v_orig, -1)  # create vector of size num_v_orig with all values -1
            idx_g = p * self.num_sample_per_proc + r  # index of subgraph
            num_v_sub = self.node_sampled[idx_g].size()  # number of nodes sampled in this subgraph
            self.ret_indptr[idx_g] = vector[int](num_v_sub + 1, 0)
            self.ret_indices[idx_g] = vector[int]()
            self.ret_indices_orig[idx_g] = vector[int]()
            self.ret_data[idx_g] = vector[float]()
            self.ret_edge_index[idx_g] = vector[int]()
            i_end = num_v_sub
            i = 0
            while i < i_end:
                _arr_bit[self.node_sampled[idx_g][i]] = i  # _arr_bit is -1 everywhere except at the org. position of
                # the nodes sampled in the subgraph, where it contains the index of the node in self.node_sampled
                i = i + 1
            i = 0
            while i < i_end:
                v = self.node_sampled[idx_g][i]  # iterate over the nodes sampled in subgraph
                start_neigh = self.adj_indptr_vec[v]  # v is index in original matrix
                end_neigh = self.adj_indptr_vec[v + 1]
                j = start_neigh  # index in original matrix at which the row elements start
                while j < end_neigh:  # iterate over ALL neighbors of the node
                    if _arr_bit[self.adj_indices_vec[j]] > -1 and self.no_rw[idx_g][v] == self.no_rw[idx_g][self.adj_indices_vec[j]]:  # if the neighbor was also sampled
                        self.ret_indices[idx_g].push_back(_arr_bit[self.adj_indices_vec[j]])  # add to self.ret_indices
                        # the index of the neighbor in self.node_sampled (will be the column)
                        self.ret_indices_orig[idx_g].push_back(self.adj_indices_vec[j])  # add to self.ret_indices_org
                        # the column of the neighbor in the original adjacency
                        self.ret_edge_index[idx_g].push_back(j)  # add to ret_edge_index the index of the neighbor in
                        # self.adj_indices_vec
                        self.ret_indptr[idx_g][_arr_bit[v] + 1] = self.ret_indptr[idx_g][_arr_bit[v] + 1] + 1
                        self.ret_data[idx_g].push_back(1.)
                    j = j + 1
                i = i + 1
            cumsum = self.ret_indptr[idx_g][0]
            i = 0
            while i < i_end:
                cumsum = cumsum + self.ret_indptr[idx_g][i + 1]
                self.ret_indptr[idx_g][i + 1] = cumsum
                i = i + 1
            r = r + 1

    def get_return(self):
        """
        Convert the subgraph related data structures from C++ to python. So that cython
        can return them to the PyTorch trainer.

        Inputs:
            None

        Outputs:
            see outputs of the `par_sample()` function.
        """
        num_subg = self.num_proc*self.num_sample_per_proc
        l_subg_indptr = []
        l_subg_indices = []
        l_subg_data = []
        l_subg_nodes = []
        l_subg_roots = []
        l_subg_edge_index = []
        offset_nodes = [0]
        offset_roots = [0]
        offset_indptr = [0]
        offset_indices = [0]
        offset_data = [0]
        offset_edge_index = [0]
        for r in range(num_subg):
            offset_nodes.append(offset_nodes[r]+self.node_sampled[r].size())
            offset_roots.append(offset_roots[r] + self.root_sampled[r].size())
            offset_indptr.append(offset_indptr[r]+self.ret_indptr[r].size())
            offset_indices.append(offset_indices[r]+self.ret_indices[r].size())
            offset_data.append(offset_data[r]+self.ret_data[r].size())
            offset_edge_index.append(offset_edge_index[r]+self.ret_edge_index[r].size())
        cdef vector[int] ret_nodes_vec = vector[int]()
        cdef vector[int] ret_roots_vec = vector[int]()
        cdef vector[int] ret_indptr_vec = vector[int]()
        cdef vector[int] ret_indices_vec = vector[int]()
        cdef vector[int] ret_edge_index_vec = vector[int]()
        cdef vector[float] ret_data_vec = vector[float]()
        ret_nodes_vec.reserve(offset_nodes[num_subg])
        ret_roots_vec.reserve(offset_roots[num_subg])
        ret_indptr_vec.reserve(offset_indptr[num_subg])
        ret_indices_vec.reserve(offset_indices[num_subg])
        ret_data_vec.reserve(offset_data[num_subg])
        ret_edge_index_vec.reserve(offset_edge_index[num_subg])
        for r in range(num_subg):
            ret_nodes_vec.insert(ret_nodes_vec.end(),self.node_sampled[r].begin(),self.node_sampled[r].end())
            ret_roots_vec.insert(ret_roots_vec.end(), self.root_sampled[r].begin(), self.root_sampled[r].end())
            ret_indptr_vec.insert(ret_indptr_vec.end(),self.ret_indptr[r].begin(),self.ret_indptr[r].end())
            ret_indices_vec.insert(ret_indices_vec.end(),self.ret_indices[r].begin(),self.ret_indices[r].end())
            ret_edge_index_vec.insert(ret_edge_index_vec.end(),self.ret_edge_index[r].begin(),self.ret_edge_index[r].end())
            ret_data_vec.insert(ret_data_vec.end(),self.ret_data[r].begin(),self.ret_data[r].end())

        cdef cutils.array_wrapper_int wint_indptr = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_int wint_indices = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_int wint_nodes = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_int wint_roots = cutils.array_wrapper_int()
        cdef cutils.array_wrapper_float wfloat_data = cutils.array_wrapper_float()
        cdef cutils.array_wrapper_int wint_edge_index = cutils.array_wrapper_int()

        wint_indptr.set_data(ret_indptr_vec)
        ret_indptr_np = np.frombuffer(wint_indptr,dtype=np.int32)
        wint_indices.set_data(ret_indices_vec)
        ret_indices_np = np.frombuffer(wint_indices,dtype=np.int32)
        wint_nodes.set_data(ret_nodes_vec)
        ret_nodes_np = np.frombuffer(wint_nodes,dtype=np.int32)
        wint_roots.set_data(ret_roots_vec)
        ret_roots_np = np.frombuffer(wint_roots, dtype=np.int32)
        wfloat_data.set_data(ret_data_vec)
        ret_data_np = np.frombuffer(wfloat_data,dtype=np.float32)
        wint_edge_index.set_data(ret_edge_index_vec)
        ret_edge_index_np = np.frombuffer(wint_edge_index,dtype=np.int32)

        for r in range(num_subg):
            l_subg_nodes.append(ret_nodes_np[offset_nodes[r]:offset_nodes[r+1]])
            l_subg_roots.append(ret_roots_np[offset_roots[r]:offset_roots[r + 1]])
            l_subg_indptr.append(ret_indptr_np[offset_indptr[r]:offset_indptr[r+1]])
            l_subg_indices.append(ret_indices_np[offset_indices[r]:offset_indices[r+1]])
            l_subg_data.append(ret_data_np[offset_data[r]:offset_data[r+1]])
            l_subg_edge_index.append(ret_edge_index_np[offset_indices[r]:offset_indices[r+1]])

        return l_subg_indptr,l_subg_indices,l_subg_data,l_subg_nodes,l_subg_edge_index, l_subg_roots

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def par_sample(self):
        """
        The main function for the sampler class. It launches multiple independent samplers
        in parallel (task parallelism by openmp), where the serial sampling function is defined
        in the corresponding sub-class. Then it returns node-induced subgraph by `_adj_extract()`,
        and convert C++ vectors to python lists / numpy arrays by `_get_return()`.

        Suppose we sample P subgraphs in parallel. Each subgraph has n nodes and e edges.

        Inputs:
            None

        Outputs (elements in the list of `ret`):
            l_subg_indptr       list of np array, length of list = P and length of each array is n+1
            l_subg_indices      list of np array, length of list = P and length of each array is m.
                                node IDs in the array are renamed to be subgraph ID (range: 0 ~ n-1)
            l_subg_data         list of np array, length of list = P and length of each array is m.
                                Normally, values in the array should be all 1.
            l_subg_nodes        list of np array, length of list = P and length of each array is n.
                                Element i in the array shows the training graph node ID of the i-th
                                subgraph node.
            l_subg_edge_index   list of np array, length of list = P and length of each array is m.
                                Element i in the array shows the training graph edge index of the
                                i-the subgraph edge.
        """
        cdef int p = 0
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc,schedule='dynamic'):
                self.sample(p)
                self.adj_extract(p)
        ret = self.get_return()
        _len = self.num_proc*self.num_sample_per_proc
        self.node_sampled.swap(vector[vector[int]](_len))
        self.root_sampled.swap(vector[vector[int]](_len))
        self.ret_indptr.swap(vector[vector[int]](_len))
        self.ret_indices.swap(vector[vector[int]](_len))
        self.ret_indices_orig.swap(vector[vector[int]](_len))
        self.ret_data.swap(vector[vector[float]](_len))
        self.ret_edge_index.swap(vector[vector[int]](_len))
        return ret

# ----------------------------------------------------

cdef class RW(Sampler):
    cdef int size_root, size_depth
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc,
                        int size_root, int size_depth):
        self.size_root = size_root
        self.size_depth = size_depth

    cdef void sample(self, int p) nogil:
        cdef int iroot = 0
        cdef int idepth = 0
        cdef int r = 0
        cdef int idx_subg
        cdef int v
        cdef int num_train_node = self.node_train_vec.size()
        while r < self.num_sample_per_proc:
            idx_subg = p*self.num_sample_per_proc+r
            # sample root
            iroot = 0
            while iroot < self.size_root:
                v = self.node_train_vec[rand()%num_train_node]
                self.node_sampled[idx_subg].push_back(v)
                # sample random walk
                idepth = 0
                while idepth < self.size_depth:
                    if (self.adj_indptr_vec[v+1]-self.adj_indptr_vec[v]>0):
                        v = self.adj_indices_vec[self.adj_indptr_vec[v]+rand()%(self.adj_indptr_vec[v+1]-self.adj_indptr_vec[v])]
                        self.node_sampled[idx_subg].push_back(v)
                    idepth = idepth + 1
                iroot = iroot + 1
            r = r + 1
            sort(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end())
            self.node_sampled[idx_subg].erase(unique(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end()),self.node_sampled[idx_subg].end())


# RW no overlap of nodes within subgraph
cdef class DisjointRW(SamplerNO):
    cdef int size_root, size_depth
    def __cinit__(self, np.ndarray[int, ndim=1, mode='c'] adj_indptr,
                  np.ndarray[int, ndim=1, mode='c'] adj_indices,
                  np.ndarray[int, ndim=1, mode='c'] node_train,
                  int num_proc, int num_sample_per_proc,
                  int size_root, int size_depth):
        self.size_root = size_root
        self.size_depth = size_depth

    cdef void sample(self, int p) nogil:
        cdef int iroot = 0
        cdef int inode = 0
        cdef int idepth = 0
        cdef int r = 0
        cdef int idx_subg
        cdef int v
        cdef int num_train_node = self.node_train_vec.size()
        cdef bool found
        cdef int neigh_size = 0
        while r < self.num_sample_per_proc:
            idx_subg = p * self.num_sample_per_proc + r
            # sample root
            iroot = 0
            remaining_root = self.node_train_vec
            while iroot < self.size_root:
                inode = rand() % remaining_root.size()
                v = remaining_root[inode]
                self.node_sampled[idx_subg].push_back(v)
                self.root_sampled[idx_subg].push_back(v)
                self.no_rw[idx_subg][v] = iroot
                remaining_root.erase(remaining_root.begin()+inode)
                # sample random walk
                idepth = 0
                while idepth < self.size_depth:
                    if self.adj_indptr_vec[v + 1] - self.adj_indptr_vec[v] > 0:
                        # new_v = self.adj_indices_vec[
                        #     self.adj_indptr_vec[v] + rand() % (self.adj_indptr_vec[v + 1] - self.adj_indptr_vec[v])]
                        # if cutils.find(self.node_sampled[idx_subg].begin(), self.node_sampled[idx_subg].end(), new_v) == self.node_sampled[idx_subg].end():
                        #     self.node_sampled[idx_subg].push_back(new_v)
                        #     self.no_rw[idx_subg][new_v] = iroot
                        #     remaining_root.erase(cutils.find(remaining_root.begin(), remaining_root.end(), new_v))
                        v = self.adj_indices_vec[
                            self.adj_indptr_vec[v] + rand() % (self.adj_indptr_vec[v + 1] - self.adj_indptr_vec[v])]
                        if cutils.find(self.node_sampled[idx_subg].begin(), self.node_sampled[idx_subg].end(), v) == \
                                self.node_sampled[idx_subg].end():
                            self.node_sampled[idx_subg].push_back(v)
                            self.no_rw[idx_subg][v] = iroot
                            remaining_root.erase(cutils.find(remaining_root.begin(), remaining_root.end(), v))
                    idepth = idepth + 1
                iroot = iroot + 1
            r = r + 1
            sort(self.node_sampled[idx_subg].begin(), self.node_sampled[idx_subg].end())
            self.node_sampled[idx_subg].erase(
                unique(self.node_sampled[idx_subg].begin(), self.node_sampled[idx_subg].end()),  # returns pointer to end of new vector
                self.node_sampled[idx_subg].end())



cdef class NodesUniformMaxDegree(MaxDegreeSampler):
    cdef int size_subgraph
    cdef int max_degree
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc,
                        int size_subgraph, int max_degree):
        self.size_subgraph = size_subgraph
        self.max_degree = max_degree

    cdef void sample(self, int p) nogil:
        cdef int inode = 0
        cdef int neigh_indices = 0
        cdef int r = 0
        cdef int idx_subg
        cdef int v
        cdef int num_train_node = self.node_train_vec.size()
        cdef vector[int] counts
        cdef int neigh
        while r < self.num_sample_per_proc:
            idx_subg = p*self.num_sample_per_proc+r
            counts = vector[int](num_train_node, 0)
            inode = 0
            while inode < self.size_subgraph:
                idx = rand()%num_train_node
                if counts[idx] == self.max_degree:
                    continue
                v = self.node_train_vec[idx]
                neigh_indices = self.adj_indptr_vec[v+1]-self.adj_indptr_vec[v]
                neigh = 0
                while neigh < neigh_indices:
                    new_v = self.adj_indices_vec[self.adj_indptr_vec[v]+neigh]
                    for i in range(num_train_node):
                        if self.node_train_vec[i] == new_v:
                            break
                    if counts[i] == self.max_degree:
                        neigh = neigh + 1
                        continue
                    if counts[idx] == self.max_degree:
                        break
                    counts[idx] = counts[idx] + 1
                    counts[i] = counts[i] + 1

                    self.neighbors[idx_subg][v].push_back(new_v)
                    self.neighbors[idx_subg][new_v].push_back(v)

                    if cutils.find(self.node_sampled[idx_subg].begin(), self.node_sampled[idx_subg].end(), v) == \
                            self.node_sampled[idx_subg].end():
                        inode = inode + 1
                        self.node_sampled[idx_subg].push_back(v)

                    if cutils.find(self.node_sampled[idx_subg].begin(), self.node_sampled[idx_subg].end(), new_v) == \
                            self.node_sampled[idx_subg].end():
                        inode = inode + 1
                    self.node_sampled[idx_subg].push_back(new_v)

                    neigh = neigh + 1

            r = r + 1
            sort(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end())
            self.node_sampled[idx_subg].erase(unique(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end()),self.node_sampled[idx_subg].end())


cdef class FullBatch(Sampler):
    def __cinit__(self, np.ndarray[int,ndim=1,mode='c'] adj_indptr,
                        np.ndarray[int,ndim=1,mode='c'] adj_indices,
                        np.ndarray[int,ndim=1,mode='c'] node_train,
                        int num_proc, int num_sample_per_proc):
        pass

    cdef void sample(self, int p) nogil:
        cdef int i = 0
        cdef int r = 0
        cdef int idx_subg
        cdef int sample
        while r < self.num_sample_per_proc:
            idx_subg = p*self.num_sample_per_proc+r
            i = 0
            while i < self.node_train_vec.size():
                sample = i
                self.node_sampled[idx_subg].push_back(self.node_train_vec[sample])
                i = i + 1
            r = r + 1
            sort(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end())
            self.node_sampled[idx_subg].erase(unique(self.node_sampled[idx_subg].begin(),self.node_sampled[idx_subg].end()),self.node_sampled[idx_subg].end())
