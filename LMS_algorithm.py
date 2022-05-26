import numpy as np
from scipy import linalg, signal


class LMS:

    def __init__(self, order=7, iterations=3000, ensemble=30, signal_complex=True):
        self.ensemble = ensemble
        self.K = iterations  # number of iterations
        self.fir_order = order
        self.N = order + 1
        self.n_coefficients = self.N
        self.n_iterations = self.K

        self.signal_complex = signal_complex

        self.rng = np.random.default_rng()

        self.d = None
        self.x = None

        if self.signal_complex:
            self.error_vector = np.zeros(self.n_iterations, dtype=complex)
            self.output_vector = np.zeros(self.n_iterations, dtype=complex)
            self.coefficient_vector = np.zeros((self.n_iterations + 1, self.n_coefficients), dtype=complex)
        else:
            self.error_vector, self.output_vector = np.zeros((2, self.n_iterations))
            self.coefficient_vector = np.zeros((self.n_iterations + 1, self.n_coefficients))

    # Implements the Complex LMS_algorithm algorithm for COMPLEX valued data.
    # (Algorithm 3.2 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    def lms(self, desired, fed, *, step, fir_order, init_coefficients):
        prefixed_input = np.concatenate((np.zeros(fir_order), fed))
        self.coefficient_vector[0] = init_coefficients

        for i in range(len(desired)):
            regress = prefixed_input[i:i + fir_order + 1]
            output = signal.lfilter(self.coefficient_vector[i].conj(), [1], regress)[-1]
            e = desired[i] - output
            coefficient_new = (self.coefficient_vector[i]
                               + 2 * step * e.conj()
                               * regress[::-1])

            self.output_vector[i] = output
            self.error_vector[i] = e
            self.coefficient_vector[i + 1] = coefficient_new

        return self.coefficient_vector, self.error_vector

    # Implements the Sign-Error LMS algorithm for REAL valued data.
    # (Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    def sign_error(self, desired, fed, *, step, fir_order, init_coefficients):
        prefixed_input = np.concatenate((np.zeros(fir_order), fed))
        self.coefficient_vector[0] = init_coefficients

        for i in range(len(desired)):
            regress = prefixed_input[i:i + fir_order + 1]
            output = signal.lfilter(self.coefficient_vector[i].conj(), [1], regress)[-1]
            e = desired[i] - output
            coefficient_new = (self.coefficient_vector[i]
                               + 2 * step * np.sign(np.real(e))
                               * regress[::-1])

            self.output_vector[i] = output
            self.error_vector[i] = e
            self.coefficient_vector[i + 1] = coefficient_new

        return self.coefficient_vector, self.error_vector

    # Implements the Dual-Sign-Error LMS algorithm for REAL valued data.
    # (Modified version of Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    def dual_sign_error(self, desired, fed, *, step, fir_order, init_coefficients):
        prefixed_input = np.concatenate((np.zeros(fir_order), fed))
        self.coefficient_vector[0] = init_coefficients

        for i in range(len(desired)):
            regress = prefixed_input[i:i + fir_order + 1]
            output = signal.lfilter(self.coefficient_vector[i].conj(), [1], regress)[-1]
            e = desired[i] - output
            dual_sign_error = e * np.sign(np.real(e)) if abs(e) > 1 else np.sign(np.real(e))
            coefficient_new = (self.coefficient_vector[i]
                               + 2 * step * dual_sign_error
                               * regress[::-1])

            self.output_vector[i] = output
            self.error_vector[i] = e
            self.coefficient_vector[i + 1] = coefficient_new

        return self.coefficient_vector, self.error_vector

    # Implements the Complex Affine-Projection algorithm for COMPLEX valued data.
    # (Algorithm 4.6 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    def affine_projection(self, desired, fed, gamma=.01, *, step, fir_order, init_coefficients, memory_length):
        prefixed_input = np.concatenate((np.zeros(fir_order), fed))
        prefixed_desired = np.concatenate((np.zeros(memory_length), desired))
        self.coefficient_vector[0] = init_coefficients
        regress = np.zeros((fir_order + 1, memory_length + 1))

        for i in range(len(desired)):
            regress, _ = np.hsplit(regress, [memory_length])
            regress = np.column_stack(((prefixed_input[i:i + fir_order + 1])[::-1], regress))
            output = regress.T.conj() @ self.coefficient_vector[i]
            e = ((prefixed_desired[i:i + memory_length + 1])[::-1]).conj() - output
            coefficient_new = (self.coefficient_vector[i]
                               + 2 * step * regress @ linalg.inv(
                        regress.T.conj() @ regress + gamma * np.eye(memory_length + 1)) @ e)

            self.output_vector[i] = output[0].conj()
            self.error_vector[i] = e[0].conj()
            self.coefficient_vector[i + 1] = coefficient_new

        return self.coefficient_vector, self.error_vector

    # Implements the momentum LMS_algorithm algorithm for COMPLEX valued data.
    def mlms(self, desired, fed, gamma=.9, *, step, fir_order, init_coefficients):
        prefixed_input = np.concatenate((np.zeros(fir_order), fed))
        self.coefficient_vector[0] = init_coefficients

        b = np.zeros(fir_order + 1)
        for i in range(len(desired)):
            regress = prefixed_input[i:i + fir_order + 1]
            output = signal.lfilter(self.coefficient_vector[i].conj(), [1], regress)[-1]
            e = desired[i] - output
            b = gamma * b + (1 - gamma) * (-2) * e.conj() * regress[::-1]
            coefficient_new = (self.coefficient_vector[i]
                               - step * b)

            self.output_vector[i] = output
            self.error_vector[i] = e
            self.coefficient_vector[i + 1] = coefficient_new

        return self.coefficient_vector, self.error_vector

    # Implements the LMS_algorithm-Newton algorithm for COMPLEX valued data.
    # (Algorithm 4.2 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    def lms_newton(self, desired, fed, alpha=.05, gamma=.1, *, step, fir_order, init_coefficients):
        prefixed_input = np.concatenate((np.zeros(fir_order), fed))
        self.coefficient_vector[0] = init_coefficients
        rx_inv_hat = gamma * np.eye(fir_order + 1)

        for i in range(len(desired)):
            regress = prefixed_input[i:i + fir_order + 1]
            output = signal.lfilter(self.coefficient_vector[i].conj(), [1], regress)[-1]
            e = desired[i] - output
            # reg_reverse = regress[::-1]
            # B = np.atleast_2d(reg_reverse).T
            # rx_hat = linalg.inv(rx_inv_hat)
            # rx_inv_hat = linalg.inv((1 - alpha) * rx_hat + alpha * B @ B.conj().T)
            rx_inv_hat = matrix_inversion_lemma(1 / (1 - alpha) * rx_inv_hat, regress[::-1], alpha, a_iteration=True)
            coefficient_new = (self.coefficient_vector[i]
                               + 2 * step * e.conj() * rx_inv_hat
                               @ regress[::-1])

            self.output_vector[i] = output
            self.error_vector[i] = e
            self.coefficient_vector[i + 1] = coefficient_new

        return self.coefficient_vector, self.error_vector

    # Implements the Normalized LMS algorithm for COMPLEX valued data.
    # (Algorithm 4.3 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    def normalized_lms(self, desired, fed, eps=1e-12, *, step, fir_order, init_coefficients):
        prefixed_input = np.concatenate((np.zeros(fir_order), fed))
        self.coefficient_vector[0] = init_coefficients

        for i in range(len(desired)):
            regress = prefixed_input[i:i + fir_order + 1]
            output = signal.lfilter(self.coefficient_vector[i].conj(), [1], regress)[-1]
            e = desired[i] - output
            coefficient_new = (self.coefficient_vector[i]
                               + step * e.conj()
                               * regress[::-1] / (eps + regress @ regress.conj()))

            self.output_vector[i] = output
            self.error_vector[i] = e
            self.coefficient_vector[i + 1] = coefficient_new

        return self.coefficient_vector, self.error_vector


def matrix_inversion_lemma(a, b, c=None, d=None, a_iteration=False):
    # linalg.inv(A + B @ C @ D)
    # if iteration use inv(A) instead of A
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c) if c is not None else np.eye(min(b.shape))
    b = b.T if b.shape[0] != a.shape[0] else b
    d = np.atleast_2d(d) if d is not None else b.conj().T
    d = d.T if b.shape == d.shape else d

    # try:
    #     det_c = abs(linalg.det(c))
    #     det_a = abs(linalg.det(a))
    #     if det_c < 1e-12 or det_a < 1e-12:
    #         raise linalg.LinAlgError('almost singular matrix')
    # except linalg.LinAlgError as e:
    #     print('LinAlgError!: A and C must be invertible')
    #     raise

    a_inv = a if a_iteration else linalg.inv(a)
    c_inv = linalg.inv(c)
    a_inv_next = a_inv - a_inv @ b @ linalg.inv(d @ a_inv @ b + c_inv) @ d @ a_inv
    return a_inv_next
