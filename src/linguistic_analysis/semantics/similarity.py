import copy
import math
import networkx as nx
import numpy as np
from typing import Callable, Iterable, List, Text

def get_discount_function_constant(value: float) -> Callable:
    return lambda index: value

def get_discount_function_exponential(rate: float) -> Callable:
    return lambda index: math.exp(-1 * rate * index)

class SemGraph():
    """
    Semantic graph. This class includes the calculation of semantic similarity between graphs.
    """

    def __init__(self, names: Iterable[Text]):
        self.graph = nx.Graph()
        assert len(set(names)) == len(names)  # Ensure no repeated names
        self.names = sorted(set(names))
        self.dimension = len(names)
        self.indexes = {name: i for i, name in enumerate(names)}
        for i, n in enumerate(names):
            self.graph.add_node(i, name=n, s=self.__get_initial_score(i))

    def __get_initial_score(self, index: int):
        return np.array([1.0 if j==index else 0.0 for j in range(self.dimension)])

    def add_edge_with_names(self, name_1:Text, name_2:Text, similarity:float):
        index_1 = self.indexes[name_1]
        index_2 = self.indexes[name_2]
        self.graph.add_edge(index_1, index_2, similarity=similarity)

    def reset(self):
        for i in range(self.dimension):
            self.graph.nodes[i]["s"] = self.__get_initial_score(i)

    def get_score_vectors(self) -> List[np.ndarray]:
        return [self.graph.nodes[i]["s"] for i in range(self.dimension)]

    def get_semantic_similarity(self, g: "SemGraph", num_iterations: int,
                                discount_function: Callable= lambda index: 0.9,
                                normalize: bool= True) -> float:
        assert self.names == g.names
        assert self.dimension == g.dimension
        self.propagate(num_iterations, discount_function)
        g.propagate(num_iterations, discount_function, normalize=normalize)
        return SemGraph.__calculate_score_vectors_distance(self.get_score_vectors(), g.get_score_vectors())

    def propagate(self, num_iterations: int, discount_function: Callable= lambda index: 0.9,
                  normalize: bool= True):
        for i in range(num_iterations):
            # Save a temporary copy of similarity vectors to use them in the next step
            for j in range(self.dimension):
                self.graph.nodes[j]["s_temp"] = copy.deepcopy(self.graph.nodes[j]["s"])
            for j in range(self.dimension):
                s_update = 0
                for n in self.graph.neighbors(j):
                    s_update += self.graph.nodes[n]["s_temp"]*self.graph[j][n]["similarity"]
                self.graph.nodes[j]["s"] += discount_function(i)*s_update
            if normalize:
                self.normalize()

    def normalize(self):
        for i in range(self.dimension):
            s = self.graph.nodes[i]["s"]
            self.graph.nodes[i]["s"] = s / np.linalg.norm(s)

    @classmethod
    def __calculate_score_vectors_distance(cls, s1: Iterable[np.ndarray], s2: Iterable[np.ndarray]) -> float:
        res = 0
        assert len(list(s1)) == len(list(s2))
        for i, s in enumerate(s1):
            res += np.linalg.norm(s1, s2[i])
        return res
