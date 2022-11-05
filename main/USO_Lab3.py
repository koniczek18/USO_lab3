import numpy as np
import scipy.signal as sig
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def gibberish():
    a = np.array([[0, 2, -3], [1, 0, -2], [0, 1, 0]])
    b = np.array([[1], [0], [0]])
    print(b)
    print(np.dot(a, b))
    print(np.dot(np.dot(a, a), b))


def kalmanMatrixSS(A, B, n):
    col1 = B
    col2 = np.dot(A, B)
    col3 = np.dot(np.dot(A, A), B)
    if n == 2:
        return np.column_stack([col1, col2])
    elif n == 3:
        return np.column_stack([col1, col2, col3])


def zad1():
    # initialise RLC parameters
    r1 = 1
    r2 = 2
    r4 = 4
    c05 = 0.5
    c1 = 1
    c2 = 2
    c3 = 3
    l05 = 0.5
    l1 = 1
    # system1 init
    A1 = np.array([[-1 / (r1 * c1), 0], [0, -1 / (r2 * c2)]])
    B1 = np.array([[1 / (r1 * c1)], [1 / (r2 * c2)]])
    C1 = np.array([[-1, 0], [0, -1]])
    D1 = np.array([[1], [1]])
    KM1 = kalmanMatrixSS(A1, B1, 2)


if __name__ == '__main__':
    zad1()
