import numpy as np
from numpy import ma as ma
from scipy import linalg
from scipy.sparse import linalg as splinalg


class L1Ball:
    """Indicator function over the L1 ball
  This function is 0 if the sum of absolute values is less than or equal to
  alpha, and infinity otherwise.
  """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        if np.abs(x).sum() <= self.alpha:
            return 0
        else:
            return np.infty

    # def prox(self, x, step_size):
    #     return euclidean_proj_l1ball(x, self.alpha)

    def lmo(self, u, x):
        """Return s - x, s solving the linear problem
            max_{||s||_1 <= alpha} <u, s>
        """
        abs_u = np.abs(u)
        largest_coordinate = np.argmax(abs_u)

        update_direction = -x.copy()
        update_direction[largest_coordinate] += self.alpha * np.sign(
            u[largest_coordinate]
        )

        return update_direction, 1
