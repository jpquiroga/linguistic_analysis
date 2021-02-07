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

    @classmethod
    def build_from_gensim_embedding(cls, embedding_model: "gensim.models.base_any2vec.BaseWordEmbeddingsModel",
                             base_words: List[Text], n_top_similar: int= 10, similarity_threshold: float= 0.2)\
            -> "SemGraph":
        """
        Build a Semgraph instance from a list of base words an and a gensim embedding model.
        :param embedding_model:
        :param base_words: List of words to be taken as the basis for build a Semgraph.
        :param n_top_similar: The number of most similar words to be analyzed for every base word.
        :param similarity_threshold: Words with lower similarity than this threshold will not be added to the Semgraph.
            Edges with lower similarity than this threshold will not be added either.

        Cosine similarity is used.

        :return: A Semgraph.
        """
        # 1. Identify words to add to the Semgraph
        words = [w for w in base_words]
        for w in base_words:
            candidate_words = embedding_model.similar_by_word(w, topn=n_top_similar)
            for w,sim in candidate_words:
                if sim > similarity_threshold and w not in words:
                    words.append(w)
        words = sorted(set(words))
        res = SemGraph(words)
        # 2 Add similarity values as edge weights
        for i in range(len(words)):
            for j in range(i, len(words)):
                w1 = words[i]
                w2 = words[j]
                if w1 != w2:
                    similarity = embedding_model.similarity(w1, w2)
                    if similarity > similarity_threshold:
                        res.add_edge_with_names(w1, w2, similarity)
        return res

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

    def save_to_gefx(self, path:Text):
        """
        Save the current graph to GEXF (Graph Exchange XML Format).
        :param path: The path to export to.
        """
        # numpy ndarrays cannot be saved to GEXF. Need to convert them to strings.
        g_exp = self.graph.copy()
        for n in g_exp.nodes:
            g_exp.nodes[n]["s"] = str(g_exp.nodes[n]["s"])
        nx.write_gexf(g_exp, path)
