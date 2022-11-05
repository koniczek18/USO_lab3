import numpy as np
import scipy.signal as sig
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def gibberish():
    a = np.array([[0, 2, -3], [1, 0, -2], [0, 1, 0]])
    b = np.array([[1], [0], [0]])
    # macierz*wektor => np.dot(m,w)
    print(b)
    print(np.dot(a, b))
    print(np.dot(np.dot(a, a), b))

def zad2():
    pass

if __name__ == '__main__':
    zad2()
