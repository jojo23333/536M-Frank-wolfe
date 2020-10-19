import numpy as np
from copy import deepcopy
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
                # print(f"loss: {fx}")
            else:
                self.trace_x.append(dl["x"].copy())
            delta = (datetime.now() - self.start).total_seconds()
            self.trace_time.append(delta)
            self.trace_step_size.append(dl["step_size"])
        self._counter += 1


def test_fw(loss_func, constraint, x0, max_iter=50):
    cb = Trace(f=loss_func)
    opt = minimize_frank_wolfe(
            loss_func,
            x0,
            constraint.lmo,
            tol=1e-3,
            callback=cb,
            max_iter=max_iter
        )
    
    assert np.isfinite(opt.x).sum() == n_features
    return cb


if __name__ == "__main__":
    np.random.seed(0)
    n_features, n_out = 400 , 20
    alpha = 0.5

    A = np.random.randn(n_out, n_features)
    b = np.random.randn(n_out)
    x0 = np.zeros(n_features)

    loss_func = SquaredLoss(A, b)

    l1ball_1 = L1Ball(alpha=0.1)
    cb_1 = test_fw(loss_func, l1ball_1, deepcopy(x0))

    l1ball_2 = L1Ball(alpha=0.4)
    cb_2 = test_fw(loss_func, l1ball_2, deepcopy(x0))

    l1ball_3 = L1Ball(alpha=0.7)
    cb_3 = test_fw(loss_func, l1ball_3, deepcopy(x0))

    l1ball_4 = L1Ball(alpha=1)
    cb_4 = test_fw(loss_func, l1ball_4, deepcopy(x0))

    l1ball_5 = L1Ball(alpha=2)
    cb_5 = test_fw(loss_func, l1ball_5, deepcopy(x0))

    from matplotlib import pyplot as plt

    # x_axix = range(1, 201)
    plt.plot(range(0, len(cb_1.trace_fx)), cb_1.trace_fx, color='green', label='t = 0.1')
    plt.plot(range(0, len(cb_2.trace_fx)), cb_2.trace_fx, color='red', label='t = 0.4')
    plt.plot(range(0, len(cb_3.trace_fx)), cb_3.trace_fx,  color='skyblue', label='t = 0.7')
    plt.plot(range(0, len(cb_4.trace_fx)), cb_4.trace_fx, color='blue', label='t = 1')
    plt.plot(range(0, len(cb_5.trace_fx)), cb_5.trace_fx, color='purple', label='t = 2')
    plt.legend(fontsize=18) # 显示图例
    plt.tick_params(labelsize=18)
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Squared Loss', fontsize=18)
    plt.show()

