import numpy as np
import scipy.signal as sig
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
import control

#Kod zawiera pakiet 'control'
#Powinno w danym momencie być aktywne jedno zadanie w mainie (inaczej wystąpią problemy z wyświetlaniem wykresów

def checkControllability(cal,n):
    if np.linalg.matrix_rank(cal)==n:
        return('jest sterowalny.')
    else:
        return('nie jest sterowalny')

def zadanie1(active):
    if active:
        #1.1 - Tak, ponieważ w przeciwieństwie do transmitancji, istnieje nieskończenie wiele rozwiązań w przestrzeni
        #      równiań stanu, które opisują ten sam system/model.
        #
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
        C1a = np.array([1, 0])
        C1b = np.array([0, 1])
        D1 = np.array([0])
        system1a = sig.lti(A1, B1, C1a, D1)
        system1b = sig.lti(A1, B1, C1b, D1)
        # system2 init
        A2 = np.array([[-1 / (r1 * c1), 0, 0], [0, -1 / (r1 * c2), 0], [0, 0, -1 / (r1 * c3)]])
        B2 = np.array([[1 / (r1 * c1)], [1 / (r1 * c2)], [1 / (r1 * c3)]])
        C2a = np.array([1, 0, 0])
        C2b = np.array([0, 1, 0])
        C2c = np.array([0, 0, 1])
        D2 = np.array([0])
        system2a = sig.lti(A2, B2, C2a, D2)
        system2b = sig.lti(A2, B2, C2b, D2)
        system2c = sig.lti(A2, B2, C2c, D2)
        # system3 init
        A3 = np.array([-1 / c1 * r1])
        B3 = np.array([0])
        C3 = np.array([1])
        D3 = np.array([0])
        system3 = sig.lti(A3, B3, C3, D3)
        # n3 =
        # system4 init
        A4 = np.array([[-r2 / l05, 0, -1 / l05], [0, 0, 1 / l1], [1 / c2, -1 / c2, -1 / (r1 * c2)]])
        B4 = np.array([[1 / l05], [0], [0]])
        C4a = np.array([1, 0, 0])
        C4b = np.array([0, 1, 0])
        C4c = np.array([0, 0, 1])
        D4 = np.array([0])
        system4a = sig.lti(A4, B4, C4a, D4)
        system4b = sig.lti(A4, B4, C4b, D4)
        system4c = sig.lti(A4, B4, C4c, D4)
        n4 = 3
        # control check
        cal1 = control.ctrb(A1, B1)
        cal2 = control.ctrb(A2, B2)
        cal3 = control.ctrb(A3, B3)
        cal4 = control.ctrb(A4, B4)
        print(cal1)
        print(cal2)
        print(cal3)
        print(cal4)
        print('System 1', checkControllability(cal1, 2))
        print('System 2', checkControllability(cal2, 3))
        print('System 3', checkControllability(cal3, 1))
        print('System 4', checkControllability(cal4, 3))
        # simulation init
        t = np.linspace(0, 10, 1001)
        step = np.ones_like(t)
        sine = np.sin(2 * np.pi * t)
        # simulate
        # system 1
        ts1a, ys1a, xs1a = sig.lsim2(system1a, step, t)
        tsine1a, ysine1a, xsine1a = sig.lsim2(system1a, sine, t)
        ts1b, ys1b, xs1b = sig.lsim2(system1b, step, t)
        tsine1b, ysine1b, xsine1b = sig.lsim2(system1b, sine, t)
        # system 2
        ts2a, ys2a, xs2a = sig.lsim2(system2a, step, t)
        tsine2a, ysine2a, xsine2a = sig.lsim2(system2a, sine, t)
        ts2b, ys2b, xs2b = sig.lsim2(system2b, step, t)
        tsine2b, ysine2b, xsine2b = sig.lsim2(system2b, sine, t)
        ts2c, ys2c, xs2c = sig.lsim2(system2c, step, t)
        tsine2c, ysine2c, xsine2c = sig.lsim2(system2c, sine, t)
        # system 3
        # pominęty, widać brak sterowalności
        # system 4
        ts4a, ys4a, xs4a = sig.lsim2(system4a, step, t)
        tsine4a, ysine4a, xsine4a = sig.lsim2(system4a, sine, t)
        ts4b, ys4b, xs4b = sig.lsim2(system4b, step, t)
        tsine4b, ysine4b, xsine4b = sig.lsim2(system4b, sine, t)
        ts4c, ys4c, xs4c = sig.lsim2(system4c, step, t)
        tsine4c, ysine4c, xsine4c = sig.lsim2(system4c, sine, t)
        # plotting
        # system 1
        plt.figure(1)
        plt.plot(t, ys1a, label='Step system 1.1')
        plt.plot(t, ysine1a, label='Sine system 1.1')
        plt.plot(t, ys1b, label='Step system 1.2')
        plt.plot(t, ysine1b, label='Sine system 1.2')
        plt.xlabel('Time')
        plt.ylabel('System 1')
        plt.legend()
        # system 2
        plt.figure(2)
        plt.plot(t, ys2a, label='Step system 2.1')
        plt.plot(t, ysine2a, label='Sine system 2.1')
        plt.plot(t, ys2b, label='Step system 2.2')
        plt.plot(t, ysine2b, label='Sine system 2.2')
        plt.plot(t, ys2c, label='Step system 2.3')
        plt.plot(t, ysine2c, label='Sine system 2.3')
        plt.xlabel('Time')
        plt.ylabel('System 2')
        plt.legend()
        # system 3
        # pominięty
        # system 4
        plt.figure(4)
        plt.plot(t, ys4a, label='Step system 4.1')
        plt.plot(t, ysine4a, label='Sine system 4.1')
        plt.plot(t, ys4b, label='Step system 4.2')
        plt.plot(t, ysine4b, label='Sine system 4.2')
        plt.plot(t, ys4c, label='Step system 4.3')
        plt.plot(t, ysine4c, label='Sine system 4.3')
        plt.xlabel('Time')
        plt.ylabel('System 4')
        plt.legend()
        plt.show()
        #1.3a - Przebiegi odpowiedzi na wymuszenia układów, które są NIEsterowalne, pokrywają się - tzn. np. wszystkie
        #       odpowiedzi skokowe są takie same, i wszystkie odpowiedzi na sygnał sin są takie same.
        #       Przebiegi dpowiedzi na wymuszenia układów sterowalnych są różne i nie pokrywają się.
        #1.3b - W przeciwieństwie do lsim, lsim2 wykorzystuje funkcję odeint to rozwiązania równania i wykonania
        #       symulacji.

def zadanie2(active):
    if active:
        #system 2
        r1 = 1
        c1 = 1
        c2 = 2
        c3 = 3
        A2 = np.array([[-1 / (r1 * c1), 0, 0], [0, -1 / (r1 * c2), 0], [0, 0, -1 / (r1 * c3)]])
        B2 = np.array([[1 / (r1 * c1)], [1 / (r1 * c2)], [1 / (r1 * c3)]])
        C2a = np.array([1, 0, 0])
        C2b = np.array([0, 1, 0])
        C2c = np.array([0, 0, 1])
        D2 = np.array([0])
        system2a = sig.lti(A2, B2, C2a, D2)
        system2b = sig.lti(A2, B2, C2b, D2)
        system2c = sig.lti(A2, B2, C2c, D2)
        A2s=np.array([[0,1,0],[0,0,1],[11/6,1,1/6]])
        B2s=np.array([[0],[0],[1]])
        # 2.1 - Nie ponieważ nie będziemy mieli wszystkich współczynników.
        systemS2a = sig.lti(A2s, B2s, C2a, D2)
        systemS2b = sig.lti(A2s, B2s, C2b, D2)
        systemS2c = sig.lti(A2s, B2s, C2c, D2)
        t = np.linspace(0, 10, 1001)
        step = np.ones_like(t)
        ta, ya, xa = sig.lsim2(system2a, step, t)
        tb, yb, xb = sig.lsim2(system2b, step, t)
        tc, yc, xc = sig.lsim2(system2c, step, t)
        tsa, ysa, xsa = sig.lsim2(systemS2a, step, t)
        tsb, ysb, xsb = sig.lsim2(systemS2b, step, t)
        tsc, ysc, xsc = sig.lsim2(systemS2c, step, t)
        plt.figure(0)
        plt.plot(t,ya,label='x1 bazowe')
        plt.plot(t, yb, label='x2 bazowe')
        plt.plot(t, yc, label='x3 bazowe')
        plt.xlabel('Time')
        plt.ylabel('System 2')
        plt.legend()
        plt.figure(1)
        plt.plot(t, ysa, label='x1 sterowalne')
        plt.plot(t, ysb, label='x2 sterowalne')
        plt.plot(t, ysc, label='x3 sterowalne')
        plt.xlabel('Time')
        plt.ylabel('System 2')
        plt.legend()
        plt.show()
        #2.2a - Tak, pownieważ nadal opisujemy ten sam obiekt, który sprowadziłby się do tej samej transmitancji
        #2.2b - Nie, zmienne stanu są inaczej opisane i przez to mają inny przebieg, przy projektowaniu UAR
        #       powinniśmy wybierać takie zmienne stanu, które zapewnią nam postać sterowalną układu, a zarazem
        #       pozwolą na logiczną (realną) interpretację zmiennych stanu (co one opisują)


def zadanie3(active):
    if active:
        pass

if __name__ == '__main__':
    zadanie1(True)
    zadanie2(False)
    #zadanie3(False)
