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
    system1 = sig.lti(A1, B1, C1, D1)
    n1 = 2
    # system2 init
    A2 = np.array([[-1 / (r1 * c1), 0, 0], [0, -1 / (r1 * c2), 0], [0, 0, -1 / (r1 * c3)]])
    B2 = np.array([[1 / (r1 * c1)], [1 / (r1 * c2)], [1 / (r1 * c3)]])
    C2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    D2 = np.array([[1], [1], [1]])
    system2 = sig.lti(A2, B2, C2, D2)
    n2 = 3
    # system3 init
    A3 = np.array([0])
    B3 = np.array([0])
    C3 = np.array([1])
    D3 = np.array([0])
    system3 = sig.lti(A3, B3, C3, D3)
    # n3 = ???
    # system4 init
    A4 = np.array([[-r2 / l05, 0, -1 / l05], [0, 0, 1 / l1], [1 / c2, -1 / c2, -1 / (r1 * c2)]])
    B4 = np.array([[1 / l05], [0], [0]])
    C4 = np.array([[0,0,0],[0,0,0],[0,0,1]])
    D4 = np.array([[0],[0],[0]])
    system4 = sig.lti(A4, B4, C4, D4)
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
    # simulation init
    t = np.linspace(0, 10, 1001)
    step = np.ones_like(t)
    impulse = np.zeros(1001)
    impulse[0] = 100000
    sine=np.sin(2*np.pi*t)
    # simulate
    # system 1
    ts1,ys1,xs1=sig.lsim2(system1,step,t)
    ti1, yi1, xi1 = sig.lsim2(system1, impulse, t)
    tsine1, ysine1, xsine1 = sig.lsim2(system1, sine, t)
    # system 2
    ts2, ys2, xs2 = sig.lsim2(system2, step, t)
    ti2, yi2, xi2 = sig.lsim2(system2, impulse, t)
    tsine2, ysine2, xsine2 = sig.lsim2(system2, sine, t)
    # system 3
    ts3, ys3, xs3 = sig.lsim2(system3, step, t)
    ti3, yi3, xi3 = sig.lsim2(system3, impulse, t)
    tsine3, ysine3, xsine3 = sig.lsim2(system3, sine, t)
    # system 4
    ts4, ys4, xs4 = sig.lsim2(system4, step, t)
    ti4, yi4, xi4 = sig.lsim2(system4, impulse, t)
    tsine4, ysine4, xsine4 = sig.lsim2(system4, sine, t)
    #plotting
    # system 1
    plt.figure(1)
    plt.plot(t,ys1,label='Step system 1',color='r')
    #plt.plot(t,yi1,label='Impulse system 1',color='b')
    plt.plot(t,ysine1,label='Sine system 1',color='k')
    plt.xlabel('Time')
    plt.ylabel('System 1')
    plt.legend()
    # system 2
    plt.figure(2)
    plt.plot(t, ys2, label='Step system 2', color='r')
    #plt.plot(t, yi2, label='Impulse system 2', color='b')
    plt.plot(t, ysine2, label='Sine system 2', color='k')
    plt.xlabel('Time')
    plt.ylabel('System 2')
    plt.legend()
    # system 3
    plt.figure(3)
    plt.plot(t, ys3, label='Step system 3', color='r')
    #plt.plot(t, yi3, label='Impulse system 3', color='b')
    plt.plot(t, ysine3, label='Sine system 3', color='k')
    plt.xlabel('Time')
    plt.ylabel('System 3')
    plt.legend()
    # system 4
    plt.figure(4)
    plt.plot(t, ys4, label='Step system 4', color='r')
    #plt.plot(t, yi4, label='Impulse system 4', color='b')
    plt.plot(t, ysine4, label='Sine system 4', color='k')
    plt.xlabel('Time')
    plt.ylabel('System 4')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    zad1()
