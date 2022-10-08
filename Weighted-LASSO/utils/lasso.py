import cvxpy as cp


def loss_fn(A, y, x):
    """
    Input:  A - Sensing Matrix
            y - Noisy Measurements
            x - Values to be reconstructed
    ||y-Ax||_{2}^{2}
    """
    return cp.norm2(A@x - y)**2

def regularizer(x):
    """
    Input:  x - Values to be reconstructed
    |x|_{1}
    """
    return cp.norm1(x)

def objective_fn(A, y, x, lambd):
    """
    ||y-Ax||_{2}^{2} + lambda * |x|_{1}
    """
    return loss_fn(A, y, x) + lambd * regularizer(x)

def mse(A, y, x):
    return (1.0 / A.shape[0]) * loss_fn(A, y, x).value
