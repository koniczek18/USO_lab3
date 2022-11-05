import numpy as np
import scipy.signal as sig
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def kalmanMatrixSS(A, B, n):
    col1 = B
    col2 = np.dot(A, B)
    col3 = np.dot(np.dot(A, A), B)
    if n == 2:
        return np.column_stack([col1, col2])
    elif n == 3:
        return np.column_stack([col1, col2, col3])


def checkControllability(KM, n):
    if np.linalg.matrix_rank(KM) == n:
        return 'jest sterowalny'
    else:
        return 'nie jest sterowalny'


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
    A1 = np.array([[-1 / (r2 * c1), 0], [0, -1 / (r4 * c05)]])
    B1 = np.array([[1 / (r2 * c1)], [1 / (r4 * c05)]])
    C1 = np.array([[-1, 0], [0, -1]])
    D1 = np.array([[1], [1]])
    n1 = 2
    # system2 init
    A2 = np.array([[-1 / (r1 * c1), 0, 0], [0, -1 / (r1 * c2), 0], [0, 0, -1 / (r1 * c3)]])
    B2 = np.array([[1 / (r1 * c1)], [1 / (r1 * c2)], [1 / (r1 * c3)]])
    C2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    D2 = np.array([[1], [1], [1]])
    n2 = 3
    # system3 init
    A3 = np.array([0])
    B3 = np.array([0])
    C3 = np.array([1])
    D3 = np.array([0])
    # n3 = ???
    # system4 init
    A4 = np.array([[-r2 / l05, 0, -1 / l05], [0, 0, 1 / l1], [1 / c2, -1 / c2, -1 / (r1 * c2)]])
    B4 = np.array([[1 / l05], [0], [0]])
    C4 = np.array([0])
    D4 = np.array([0])
    n4 = 3
    # task 1.2
    KM1 = kalmanMatrixSS(A1, B1, n1)
    print('System 1', checkControllability(KM1, n1))
    KM2 = kalmanMatrixSS(A2, B2, n2)
    print('System 2', checkControllability(KM2, n2))
    # KM3 nie wiem
    print('System 3 jest ???')
    KM4 = kalmanMatrixSS(A4, B4, n4)
    print('System 4', checkControllability(KM4, n4))


if __name__ == '__main__':
    zad1()
