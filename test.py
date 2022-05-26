import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, signal
from unittest2 import TestCase

from LMS_algorithm import LMS, matrix_inversion_lemma
rng = np.random.default_rng()


class TestLMS(TestCase):

    # noinspection PyPep8Naming
    def test_lemma(self):
        A = linalg.toeplitz([0.8 ** i for i in range(3)])
        print(A)
        B = np.ones(3)
        D = np.arange(3)
        B1, D1 = np.atleast_2d(B), np.atleast_2d(D)
        print(B1.shape)
        print(D1.shape)
        e1 = matrix_inversion_lemma(A, B, 1, D1) - linalg.inv(A + B1.T @ D1)
        print(e1)
        e2 = matrix_inversion_lemma(linalg.inv(A), B, 1, D1, True) - linalg.inv(A + B1.T @ D1)
        print(e2)
        print(e1 == e2)

    def test_ap(self):
        pi = np.pi
        sigma2_v = 1
        sigma2_n = 1e-4
        a = np.array([1, 0])
        N = 12
        mu_max = 1 / 1 * sigma2_v * np.sqrt(pi * sigma2_n / 2)
        K = 3000  # number of iterations
        print('the mu value: %.3f' % mu_max)

        mse = []
        wk_rec = []

        for j in range(50):
            u = np.sqrt(sigma2_v) * rng.standard_normal(K)
            x = u
            d = signal.lfilter([1], a, x) + np.sqrt(sigma2_n) * rng.standard_normal(K)
            ex = LMS(N - 1, signal_complex=False)
            wk, ek = ex.affine_projection(d, x,
                                          **{'fir_order': N - 1, 'init_coefficients': np.ones(N), 'step': mu_max * 0.2,
                                             'memory_length': 3})

            mse.append(abs(ek) ** 2)
            wk_rec.append(wk)

        wk_avg = np.mean(wk_rec, axis=0)[:K]
        mse_avg = np.mean(mse, axis=0)[:K]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        e2db_avg = 10 * np.log10(np.where(mse_avg == 0, float('inf'), mse_avg))
        ax1.plot(e2db_avg)
        ax2.plot(np.real(wk_avg))
        ax2.grid(color='lightgrey', linestyle='-', linewidth=1)
        plt.tight_layout()
        ax1.set_title('Learning Curve for MSE')
        ax1.set_xlabel('Number of iterations, k')
        ax1.set_ylabel('MSE [dB]')
        ax1.set_xlim(0, K)
        ax2.set_xlim(0, K)
        plt.show()
        print(wk_avg[-1])
