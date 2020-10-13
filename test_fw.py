import numpy as np
from datetime import datetime

from FW.constraint import L1Ball
from FW.frank_wolfe import minimize_frank_wolfe
from sklearn.utils.extmath import safe_sparse_dot

class SquaredLoss:
    def __init__(self, A, b, alpha=0):
        self.A = A
        self.b = b

    def __call__(self, x):
        z = safe_sparse_dot(self.A, x, dense_output=True).ravel() - self.b
        return 0.5 * (z*z).mean()
    

class Trace:
    def __init__(self, f=None, freq=1):
        self.trace_x = []
        self.trace_time = []
        self.trace_fx = []
        self.trace_step_size = []
        self.start = datetime.now()
        self._counter = 0
        self.freq = int(freq)
        self.f = f

    def __call__(self, dl):
        if self._counter % self.freq == 0:
            if self.f is not None:
                fx = self.f(dl["x"])
                self.trace_fx.append(fx)
                print(f"loss: {fx}")
            else:
                self.trace_x.append(dl["x"].copy())
            delta = (datetime.now() - self.start).total_seconds()
            self.trace_time.append(delta)
            self.trace_step_size.append(dl["step_size"])
        self._counter += 1


def test_fw(loss_func, constraint, x0):
    cb = Trace(f=loss_func)
    opt = minimize_frank_wolfe(
            loss_func,
            x0,
            constraint.lmo,
            tol=1e-3,
            callback=cb
        )
    
    assert np.isfinite(opt.x).sum() == n_features
    return cb


if __name__ == "__main__":
    n_features, n_out = 20 , 10
    alpha = 1

    A = np.random.randn(n_out, n_features)
    b = np.random.randn(n_out)
    x0 = np.zeros(n_features)

    loss_func = SquaredLoss(A, b)
    l1ball = L1Ball(alpha=alpha)

    cb = test_fw(loss_func, l1ball, x0)
    
    from matplotlib import pyplot as plt
    plt.plot(cb.trace_fx)
    plt.show()

