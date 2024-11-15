# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp import utils
from pygsp.graphs import Graph  # prevent circular import in Python < 3.5


class RandomRegular(Graph):
    """Random k-regular graph generator.

    Generates a random regular graph where every node is connected to exactly k other nodes.
    The graph is simple (no self-loops or multiple edges), k-regular (each vertex has 
    degree k), and undirected.

    Args:
        N (int): Number of nodes (default: 64)
        k (int): Number of connections per node (default: 6)
        maxIter (int): Maximum number of iterations (default: 10)
        seed (int, optional): Random seed for reproducibility
        **kwargs: Additional arguments passed to Graph constructor

    Notes:
        Uses the pairing model algorithm:
        1. Create N*k half-edges
        2. Repeatedly pick random pairs of half-edges
        3. If the pair is legal (no loops/double edges), add it to graph
        4. Continue until no half-edges remain or max iterations reached

    References:
        Kim and Vu (2003). Generating random regular graphs.
    """

    def __init__(self, N=64, k=6, maxIter=10, seed=None, **kwargs):
        self.k = k

        self.logger = utils.build_logger(__name__)

        rs = np.random.RandomState(seed)

        # continue until a proper graph is formed
        if (N * k) % 2 == 1:
            raise ValueError("input error: N*d must be even!")

        # a list of open half-edges
        U = np.kron(np.ones(k), np.arange(N)).astype(int)

        # the graphs adjacency matrix
        A = sparse.lil_matrix(np.zeros((N, N)))

        edgesTested = 0
        repetition = 1

        while np.size(U) and repetition < maxIter:
            edgesTested += 1

            # print(progess)
            if edgesTested % 5000 == 0:
                self.logger.debug("createRandRegGraph() progress: edges= "
                                  "{}/{}.".format(edgesTested, N*k/2))

            # chose at random 2 half edges
            i1 = rs.randint(0, np.shape(U)[0])
            i2 = rs.randint(0, np.shape(U)[0])
            v1 = U[i1]
            v2 = U[i2]

            # check that there are no loops nor parallel edges
            if v1 == v2 or A[v1, v2] == 1:
                # restart process if needed
                if edgesTested == N*k:
                    repetition = repetition + 1
                    edgesTested = 0
                    U = np.kron(np.ones(k), np.arange(N))
                    A = sparse.lil_matrix(np.zeros((N, N), dtype=int))
            else:
                # add edge to graph
                A[v1, v2] = 1
                A[v2, v1] = 1

                # remove used half-edges
                v = sorted([i1, i2])
                U = np.concatenate((U[:v[0]], U[v[0] + 1:v[1]], U[v[1] + 1:]))

        super(RandomRegular, self).__init__(W=A, gtype="random_regular",
                                            **kwargs)

        self.is_regular()

    def is_regular(self):
        """Validates that the generated graph is a proper regular graph.
        
        Checks for:
        - Symmetry
        - No parallel edges
        - Consistent degree (k-regular)
        - No self-loops
        
        Warns if any of these properties are violated.
        """
        warn = False
        msg = 'The given matrix'

        # check symmetry
        if np.abs(self.A - self.A.T).sum() > 0:
            warn = True
            msg = '{} is not symmetric,'.format(msg)

        # check parallel edged
        if self.A.max(axis=None) > 1:
            warn = True
            msg = '{} has parallel edges,'.format(msg)

        # check that d is d-regular
        if np.min(self.d) != np.max(self.d):
            warn = True
            msg = '{} is not d-regular,'.format(msg)

        # check that g doesn't contain any self-loop
        if self.A.diagonal().any():
            warn = True
            msg = '{} has self loop.'.format(msg)

        if warn:
            self.logger.warning('{}.'.format(msg[:-1]))
