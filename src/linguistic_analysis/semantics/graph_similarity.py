import numpy as np
from typing import Dict, List, Iterable, Text, Union

from linguistic_analysis.semantics.constants import NAME_SEPARATOR


class Triangle(object):
    """
    Triangle.
    """

    def __init__(self, ab: float, bc: float, ac: float, a_name: Text, b_name: Text, c_name: Text):
        """
        Create a triangle from the length of its edges.

        :param ab: Length of segment AB.
        :param bc: Length of segment BC.
        :param ac: Length of segment AC.
        :param a_name: The name of the vertex A.
        :param b_name: The name of the vertex B.
        :param c_name: The name of the vertex C.
        """
        assert ab > 0 and bc > 0 and ac > 0
        self.a_name = a_name
        self.b_name = b_name
        self.c_name = c_name
        # Calculate angles
        self.ab = ab
        self.bc = bc
        self.ac = ac
        self.cos_a = (ac ** 2 + ab ** 2 - bc ** 2) / (2 * ac * ab)
        self.cos_b = (bc ** 2 + ab ** 2 - ac ** 2) / (2 * ab * bc)
        self.cos_c = (bc ** 2 + ac ** 2 - ab ** 2) / (2 * ac * bc)
        self._sorted_vnames = sorted([self.a_name, self.b_name, self.c_name])
        self._triangle_name = NAME_SEPARATOR.join(self._sorted_vnames)
        # Dictionary of values
        self.d_angles: Dict[Text, float] = {self.a_name: self.cos_a, self.b_name: self.cos_b, self.c_name: self.cos_c}
        self.d_distances: Dict[Text, float] = {NAME_SEPARATOR.join(sorted([self.a_name, self.b_name])): self.ab,
                            NAME_SEPARATOR.join(sorted([self.b_name, self.c_name])): self.bc,
                            NAME_SEPARATOR.join(sorted([self.a_name, self.c_name])): self.ac}
    @property
    def name(self) -> Text:
        return self._triangle_name

    @property
    def sorted_vnames(self) -> List[str]:
        return self._sorted_vnames

    def get_distance(self, v1: Text, v2: Text) -> float:
        assert v1 in self.d_distances
        assert v2 in self.d_distances
        return self.d_distances[NAME_SEPARATOR.join(sorted([v1, v2]))]

    def _get_angle_vector(self) -> np.ndarray:
        return np.array([self.cos_a, self.cos_b, self.cos_c])

    def get_angle_distance(self, t: "Triangle", ord: Union[int, Text]=None) -> float:
        """
        Get the angle distance with another triangle, as the norm of the difference of the cosines of their angles.
        :param t: The triangle to compare with.
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

        :return:
        """
        return np.linalg.norm(self._get_angle_vector() - t._get_angle_vector(), ord=ord)

    def __str__(self):
        return f"Triangle[{self.a_name}-{self.b_name}: {self.ab}, " \
               f"{self.b_name}-{self.c_name}: {self.bc} ," \
               f"{self.a_name}-{self.c_name}: {self.ac}; " \
               f"cos_{self.a_name}: {self.cos_a}, " \
               f"cos_{self.b_name}: {self.cos_b}, " \
               f"cos_{self.c_name}:{self.cos_c}]"


class Triangulation(object):

    def __init__(self, triangles: Iterable[Triangle]):
        """
        :param triangles:
        """
        self.triangles: Iterable[Triangle] = triangles

    def get_angle_distance(self, t: "Triangulation") -> float:
        assert(len(self.triangles) == len(t.triangles))
        res = 0
        for t_1, t_2 in zip(self.triangles, t.triangles):
            res += t_1.get_angle_distance(t_2)
        return res


# TODO Move to unit tests
if __name__ == "__main__":
    import math
    t1 = Triangle(1, 1, 1, "A", "B", "C")
    print(t1)
    t2 = Triangle(1, 1, math.sqrt(2), "A", "B", "C")
    print(t2)
    t3 = Triangle(1, 100000, 100000, "A", "B", "C")
    print(t3)
    print(t1.get_angle_distance(t2))
    print(t1.get_angle_distance(t3))
    print(t2.get_angle_distance(t3))