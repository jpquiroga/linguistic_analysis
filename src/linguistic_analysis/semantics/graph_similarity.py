import numpy as np
from typing import Union, Text


class Triangle(object):

    def __init__(self, p_a: np.ndarray, p_b: np.ndarray, p_c: np.ndarray):
        assert p_a.shape == p_b.shape == p_c.shape
        self.p_a = p_a
        self.p_b = p_b
        self.p_c = p_c
        # Calculate angles
        ab = np.linalg.norm(p_a - p_b)
        bc = np.linalg.norm(p_b - p_c)
        ac = np.linalg.norm(p_c - p_a)
        self.cos_a = (ac**2 + ab**2 - bc**2)/(2*ac*ab)
        self.cos_b = (bc**2 + ab**2 - ac**2)/(2*ab*bc)
        self.cos_c = (bc**2 + ac**2 - ab**2)/(2*ac*bc)

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
        return f"Triangle[p_a: {self.p_a}, p_b: {self.p_b}, p_c: {self.p_c}; " \
               f"cos_a: {self.cos_a}, cos_b: {self.cos_b}, cos_c:{self.cos_c}]"


if __name__ == "__main__":
    import math
    t1 = Triangle(np.array([0, 0]), np.array([1, 0]), np.array([0.5, math.sqrt(3) / 2]))
    print(t1)
    t2 = Triangle(np.array([0, 0]), np.array([1, 0]), np.array([1, 1]))
    print(t2)
    print(t1.get_angle_distance(t2))
