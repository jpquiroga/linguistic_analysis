import copy
from itertools import combinations
import math
import networkx as nx
import numpy as np
from tqdm import tqdm
from typing import Callable, Iterable, List, Text, Tuple, Union

from linguistic_analysis.semantics.graph_similarity import Triangle, Triangulation, NormalizationType
from linguistic_analysis.semantics.constants import NAME_SEPARATOR


def get_discount_function_constant(value: float) -> Callable:
    return lambda index: value

def get_discount_function_exponential(rate: float) -> Callable:
    return lambda index: math.exp(-1 * rate * index)

class SemGraph():
    """
    Semantic graph. This class includes the calculation of semantic similarity between graphs.

    Every node in the graph is a word.
    Edges (those existing) have similarity values in [0, 1]. Distances are calculated as `d = 1 -s` an are also in
    [0 ,1].
    """

    def __init__(self, names: Iterable[Text]):
        self.graph = nx.Graph()
        assert len(set(names)) == len(names)  # Ensure no repeated names
        self.names = sorted(set(names))
        self.dimension = len(names)
        self.indexes = {name: i for i, name in enumerate(self.names)}
        for i, n in enumerate(self.names):
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
            if w in embedding_model:
                candidate_words = embedding_model.similar_by_word(w, topn=n_top_similar)
                for w, sim in candidate_words:
                    if sim > similarity_threshold and w not in words:
                        words.append(w)
        words = sorted(set(words))
        res = SemGraph(words)
        # 2. Add similarity values as edge weights
        for i in range(len(words)):
            for j in range(i, len(words)):
                w1 = words[i]
                w2 = words[j]
                if w1 != w2 and w1 in embedding_model and w2 in embedding_model:
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

    def set_edge_similarity(self, name_1: Text, name_2: Text, similarity:float):
        self.add_edge_with_names(name_1, name_2, similarity)

    def get_edge_similarity(self, name_1: Text, name_2: Text) -> float:
        """
        Similarity value is in [0, 1]

        :param name_1:
        :param name_2:
        :return: The similarity value.
        """
        index_1 = self.indexes[name_1]
        index_2 = self.indexes[name_2]
        if index_2 in self.graph[index_1]:
            return self.graph[index_1][index_2]["similarity"]
        else:
            return 0.0

    def get_edge_distance(self, name_1: Text, name_2: Text) -> float:
        """
        Distance value is in [0, 1]

        :param name_1:
        :param name_2:
        :return: The distance value.
        """
        return 1 - self.get_edge_similarity(name_1, name_2)

    def reset(self):
        for i in range(self.dimension):
            self.graph.nodes[i]["s"] = self.__get_initial_score(i)

    def get_score_vectors(self) -> List[np.ndarray]:
        return [self.graph.nodes[i]["s"] for i in range(self.dimension)]

    def get_semantic_distance(self, g: "SemGraph", num_iterations: int,
                              discount_function: Callable= lambda index: 0.9,
                              normalize: bool= True) -> float:
        """
        Distance is calculated on semgraphs composed by the same nodes. If the sets of nodes are different, an
        exception is raised.

        :param g: The sem graph to compare with.
        :param num_iterations:  The number of iterations to be used in the propagation algorithm. If 0, no propagation
            will be carried out.
        :param discount_function: Discount function to be used.
        :param normalize: Whether to normalize the
        :return: The distance value.
        """
        assert self.names == g.names
        assert self.dimension == g.dimension
        if num_iterations > 0:
            self.propagate(num_iterations, discount_function=discount_function, normalize=normalize)
            g.propagate(num_iterations, discount_function=discount_function, normalize=normalize)
        return SemGraph.__calculate_score_vectors_distance(self.get_score_vectors(), g.get_score_vectors())

    @classmethod
    def get_relative_semantic_distances(cls, reference_semgraph: "SemGraph",
                                        semgraphs: Iterable["SemGraph"],
                                        reset: bool,
                                        num_iterations: int,
                                        discount_function: Callable= lambda index: 0.9,
                                        normalize: bool= True) -> List[float]:
        """
        Get a list of semantic distances from a reference semgraph to a list of other semgraphs.
        All the semgraphs are supposed to have the same names for their nodes and the same dimensions.
        :param reference_semgraph: The reference semgraph from which distances will be calculated.
        :param semgraphs: Iterable with semgraphs.
        :param reset: Whether to reset the semgraphs to compare.
        :param num_iterations:  The number of iterations to be used in the propagation algorithm. If 0, no propagation
            will be carried out.
        :param discount_function: Discount function to be used.
        :param normalize: Whether to normalize the
        :return: A list with the semantic distances in the same order as in semgraphs.
        """
        for sg in semgraphs:
            assert reference_semgraph.names == sg.names
            assert reference_semgraph.dimension == sg.dimension
        res = []
        if reset:
            reference_semgraph.reset()
        if num_iterations > 0:
            reference_semgraph.propagate(num_iterations, discount_function=discount_function, normalize=normalize)
        for sg in tqdm(semgraphs):
            if reset:
                sg.reset()
            if num_iterations > 0:
                sg.propagate(num_iterations, discount_function=discount_function, normalize=normalize)
            res.append(SemGraph.__calculate_score_vectors_distance(reference_semgraph.get_score_vectors(),
                                                                   sg.get_score_vectors()))
        return res

    @classmethod
    def get_relative_semantic_distance_matrix(cls, semgraphs: Iterable["SemGraph"],
                                              reset: bool,
                                              num_iterations: int,
                                              discount_function: Callable= lambda index: 0.9,
                                              normalize: bool= True) -> np.ndarray:
        """
        Get a matrix with the semantic distances of a list of semgraphs.
        All the semgraphs are supposed to have the same names for their nodes and the same dimensions.
        :param semgraphs: Iterable with semgraphs.
        :param reset: Whether to reset the semgraphs to compare.
        :param num_iterations:  The number of iterations to be used in the propagation algorithm. If 0, no propagation
            will be carried out.
        :param discount_function: Discount function to be used.
        :param normalize: Whether to normalize the
        :return: A matrix with the relative semantic distance between pairs of semgraphs. This is a squared symmetric
            matrix with the same indexes as the `semgraphs` parameter.
        """
        res = np.zeros((len(semgraphs), len(semgraphs)))
        for sg in tqdm(semgraphs):
            if reset:
                sg.reset()
            if num_iterations > 0:
                sg.propagate(num_iterations, discount_function=discount_function, normalize=normalize)
        for i, sg_i in tqdm(enumerate(semgraphs)):
            for j, sg_j in enumerate(semgraphs):
                assert sg_i.names == sg_j.names
                res[i, j] = SemGraph.__calculate_score_vectors_distance(sg_i.get_score_vectors(),
                                                                        sg_j.get_score_vectors())
        return res

    @classmethod
    def get_union_of_graph_names(cls, semgraphs: Iterable["SemGraph"]) -> List[str]:
        res = set()
        for sg in semgraphs:
            res = res.union(set(sg.names))
        return sorted(res)

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

    def copy(self) -> "SemGraph":
        res = SemGraph(self.names)
        res.graph = copy.deepcopy(self.graph)
        res.dimension = self.dimension
        res.indexes = copy.deepcopy(self.indexes)
        return res

    def get_augmented_graph(self, additional_names: Iterable[Text]) -> "SemGraph":
        """
        Get an augmented sem graph resulting from adding a list of additional nodes.

        :param additional_names: The names of the nodes to be added.

        :return:
        """
        names = list(set(self.names).union(set(additional_names)))
        res = SemGraph(names)
        for i in range(len(self.names)):
            for j in range(i, len(self.names)):
                if self.names[i] != self.names[j]:
                    similarity = self.get_edge_similarity(self.names[i], self.names[j])
                    if similarity > 0:
                        res.add_edge_with_names(self.names[i], self.names[j], similarity)
        return res

    @classmethod
    def __calculate_score_vectors_distance(cls, s1: Iterable[np.ndarray], s2: Iterable[np.ndarray]) -> float:
        """
        Calculate the mean distance between two list of arrays. Euclidean distance is used.
        :param s1:
        :param s2:
        :return: The mean distance.
        """
        res = 0
        assert len(list(s1)) == len(list(s2))
        for i, s in enumerate(s1):
            res += np.linalg.norm(s - s2[i])
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

    def clone(self) -> "SemGraph":
        res = SemGraph(self.names)
        for i in range(len(self.names)):
            for j in range(i, len(self.names)):
                w1 = self.names[i]
                w2 = self.names[j]
                if w1 != w2:
                    similarity = self.get_edge_similarity(w1, w2)
                    if similarity > 0:
                        res.add_edge_with_names(w1, w2, similarity)
        return res

    def get_stacked_vectors(self) -> np.ndarray:
        return np.array(self.get_score_vectors()).reshape(1, -1)[0]

    @classmethod
    def save_stacked_semgraphs_to_tsv(cls, semgraphs: Iterable["SemGraph"], labels: Iterable[str], path: str):
        """
        Save the stacked representation of a list of semgraphs to a TSV format with metadata.
        :param semgraphs: The semgraphs to save.
        :param labels: The list of labels to use as metadata, in the same order as semgraphs.
        :param path: The file to write the data in TSV format. The metadata will be written to a file with the same path
            and name, with the additional suffix ".metadata".
        """
        with open(path, "w") as f:
            for sg in tqdm(semgraphs):
                f.write("\t".join([str(d) for d in sg.get_stacked_vectors()]) + "\n")
        with open(path + ".metadata", "w") as f:
            for l in tqdm(labels):
                f.write(l + "\n")

    def get_triangulation(self, normalization: NormalizationType= None) -> Triangulation:
        """
        Get a triangulation from the current semgraph.
        The triangles follow a lexicographic order, where the name of each triangle is the lexicographically ordered
        triplet of its vertex names. Example: ABC, ACD, BCD, BDE

        :param normalization: The type of normalization to be applied to the triangles of the triangulation.

        :return: The generated triangulation.
        """
        # 1. Get all the combinations of vertex triplets.
        # Only include valid triplets.
        vertex_triplets = self.__get_all_valid_vertex_triplets()
        # 2. Create triangles.
        triangles = [self.__build_triangle(triplet, normalization=normalization) for triplet in vertex_triplets]
        # 3. Lexicographically order triangles by name.
        ordered_triangles = sorted(triangles, key=lambda t: t.name)
        return Triangulation(ordered_triangles)

    def __build_triangle(self, triplet: Iterable[Text], normalization: NormalizationType= None) -> Triangle:
        a_name, b_name, c_name = triplet
        ab = self.get_edge_distance(a_name, b_name)
        bc = self.get_edge_distance(b_name, c_name)
        ac = self.get_edge_distance(a_name, c_name)
        res = Triangle(ab, bc, ac, a_name, b_name, c_name, normalization=normalization)
        return res

    def __get_all_valid_vertex_triplets(self) -> Iterable[Tuple[Text, Text, Text]]:
        all_vertex_triplets = combinations(self.names, 3)
        res = [t for t in all_vertex_triplets if self.__is_valid_vertex_triplet(t)]
        return res

    def __is_valid_vertex_triplet(self, triplet: Iterable[Text]) -> bool:
        num_vertexes_in_graph = 0
        a = triplet[0]
        b = triplet[1]
        c = triplet[2]
        if self.__is_vertex_in_graph(a, b):
            num_vertexes_in_graph += 1
        if self.__is_vertex_in_graph(b, c):
            num_vertexes_in_graph += 1
        if self.__is_vertex_in_graph(a, c):
            num_vertexes_in_graph += 1
        return num_vertexes_in_graph > 0

    def __is_vertex_in_graph(self, node_name_1: Text, node_name_2: Text) -> bool:
        index_1 = self.indexes.get(node_name_1, -1)
        index_2 = self.indexes.get(node_name_2, -1)
        return index_1 >= 0 and index_2 >= 0 and \
               ((index_1 in self.graph and index_2 in self.graph[index_1]) or
                (index_2 in self.graph and index_1 in self.graph[index_2]))

def get_triangulation_angle_distance(g1: "Semgraph", g2: "Semgraph", ord: Union[int, Text]=None, rad: bool= False,
                                     trinagle_normalization: NormalizationType= None) -> Tuple[float, float]:
    """
    Calculate the angle distance between two SemGraphs.
    :param g1:
    :param g2:
    :param ord {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object. The default is None.

            The following norms can be calculated:

            =====  ============================  ==========================
            ord    norm for matrices             norm for vectors
            =====  ============================  ==========================
            None   Frobenius norm                2-norm
            'fro'  Frobenius norm                --
            'nuc'  nuclear norm                  --
            inf    max(sum(abs(x), axis=1))      max(abs(x))
            -inf   min(sum(abs(x), axis=1))      min(abs(x))
            0      --                            sum(x != 0)
            1      max(sum(abs(x), axis=0))      as below
            -1     min(sum(abs(x), axis=0))      as below
            2      2-norm (largest sing. value)  as below
            -2     smallest singular value       as below
            other  --                            sum(abs(x)**ord)**(1./ord)
            =====  ============================  ==========================

            The Frobenius norm is given by [1]_:

                :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

            The nuclear norm is the sum of the singular values.

            Both the Frobenius and nuclear norm orders are only defined for
            matrices and raise a ValueError when ``x.ndim != 2``.

            References
            ----------
            .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
                   Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
    :param rad: If True return the angles in radians. Else, return the cosine of the angles.
    :param trinagle_normalization: The type of normalization to be applied to the triangles of the triangulations of the
        semgraphs.

    :return: (<no_normalized_distance>, <mormalized_distance>)
    """
    # 1. G1 -> G1*, G2 -> G2* (G1*: extended graph with those nodes in G2 that are not contained in G1)
    g1_e = g1.get_augmented_graph(g2.names)
    g2_e = g2.get_augmented_graph(g1.names)

    # 2. Compare triangulations
    t1 = g1_e.get_triangulation(normalization=trinagle_normalization)
    t2 = g2_e.get_triangulation(normalization=trinagle_normalization)
    return t1.get_angle_distance(t2, ord=ord, rad=rad)
