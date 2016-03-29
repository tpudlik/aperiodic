
import unittest
import numpy as np
from numpy.testing import assert_allclose

from src.dnls import dnls_rhs, fibonacci


# class TestCentralAmplitude(unittest.TestCase):

#     def test_time_zero(self):

class TestDnlsRhs(unittest.TestCase):

    def test_dimension(self):
        M = 6
        f = dnls_rhs(M, 1.23)

        actual = f(0.4, np.arange(M)).shape
        expected = (M,)

        assert actual == expected

    # def test_values(self):

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
