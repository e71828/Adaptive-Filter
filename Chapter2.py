import numpy as np
from scipy import linalg

np.set_printoptions(precision=5, suppress=True)

if __name__ == '__main__':
    Ra = 1 / 4 * np.asarray(linalg.toeplitz([4, 3, 2, 1]))
    pa = np.array([1 / 2, 3 / 8, 2 / 8, 1 / 8])
    R = [Ra]
    p = [pa]

    Rb = np.asarray(linalg.toeplitz([1, 0.8, 0.64, 0.512]))
    pb = 1 / 4 * np.array([0.4096, 0.512, 0.64, 0.8])
    R.append(Rb)
    p.append(pb)

    Rc = 1 / 3 * np.asarray(linalg.toeplitz([3, -2, 1]))
    pc = np.array([-2, 1, -1 / 2])
    R.append(Rc)
    p.append(pc)

    for Ra, pa in zip(R, p):
        print(Ra)
        print('--------------')
        print(pa)
        print('--------------')

    # Wiener solution
    w0 = []
    for Ra, pa in zip(R, p):
        wa = linalg.inv(Ra) @ pa
        w0.append(wa)
    print(w0)


    def gradient_descent(pa, Ra, mu=1 / 5):
        w1 = np.zeros(pa.shape)
        g = -2 * pa + 2 * Ra @ w1
        while linalg.norm(g) > 1e-6:
            w1 = w1 - mu * g
            g = -2 * pa + 2 * Ra @ w1
        return w1


    for Ra, pa in zip(R, p):
        D, V = linalg.eig(Ra)
        w1 = []
        w = gradient_descent(pa, Ra, mu=1 / (abs(D.max(0)) + 0.1))
        w1.append(w)
        print(w1)
