
import unittest
import numpy as np
from numpy.testing import assert_allclose

from src.dnls import dnls_rhs, fibonacci


class TestDnlsRhs(unittest.TestCase):

    def test_dimension(self):
        M = 6
        f = dnls_rhs(M, 1.23)

        actual = f(0.4, np.arange(M)).shape
        expected = (M,)

        assert actual == expected

    def test_values_periodic_uniform(self):
        M = 6
        L = 2.3
        f = dnls_rhs(M, L)

        actual = f(2.3, np.ones(shape=(M,)))
        expected = (-1j*L + 1j)*np.ones(shape=(M,))
        expected[0] = -1j*L + 0.5j
        expected[M-1] = -1j*L + 0.5j

        assert_allclose(actual, expected)


class TestFibonacci(unittest.TestCase):

    def test_base_case_1(self):
        assert fibonacci(1, 'a', 'b') == ('a',)

    def test_base_case_2(self):
        assert fibonacci(2, 'a', 'b') == ('a', 'b')

    def test_longer_chain(self):
        expected = tuple("abaababaabaab")
        assert fibonacci(len(expected), 'a', 'b') == expected

    def test_truncation(self):
        # The length of the expected tuple is not a Fibonacci number.
        expected = tuple("abaababaabaa")
        assert fibonacci(len(expected), 'a', 'b') == expected
