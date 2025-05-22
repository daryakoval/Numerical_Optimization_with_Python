import numpy as np

def quadratic_1(x: np.ndarray, need_hessian: bool = False):
    """
    Quadratic function with Q = [[1, 0], [0, 1]] (circular contours)
    """
    Q = np.array([[1, 0], [0, 1]])
    f = float(x.T @ Q @ x)
    g = 2 * Q @ x
    h = 2 * Q if need_hessian else None
    return f, g, h

def quadratic_2(x: np.ndarray, need_hessian: bool = False):
    """
    Quadratic function with Q = [[1, 0], [0, 100]] (axis-aligned elliptical contours)
    """
    Q = np.array([[1, 0], [0, 100]])
    f = float(x.T @ Q @ x)
    g = 2 * Q @ x
    h = 2 * Q if need_hessian else None
    return f, g, h

def quadratic_3(x: np.ndarray, need_hessian: bool = False):
    """
    Quadratic function with rotated elliptical contours
    """
    R = np.array([[np.sqrt(3)/2, -0.5], 
                  [0.5, np.sqrt(3)/2]])
    D = np.array([[100, 0], [0, 1]])
    Q = R.T @ D @ R
    
    f = float(x.T @ Q @ x)
    g = 2 * Q @ x
    h = 2 * Q if need_hessian else None
    return f, g, h

def rosenbrock(x: np.ndarray, need_hessian: bool = False):
    """
    f(x) = 100(x₂ - x₁²)² + (1 - x₁)²
    """
    x1, x2 = x[0], x[1]
    f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    
    g = np.array([
        -400 * x1 * (x2 - x1**2) - 2 * (1 - x1),
        200 * (x2 - x1**2)
    ])

    if need_hessian:
        h = np.array([
            [-400 * (x2 - 3*x1**2) + 2, -400 * x1],
            [-400 * x1, 200]
        ])
    else:
        h = None
    
    return f, g, h

def linear(x: np.ndarray, need_hessian: bool = False):
    """
    f(x) = a.T @ x
    Using a = [1, 2] for this example
    """
    a = np.array([3.0, 4.0])
    f = float(a @ x)
    g = a
    h = np.zeros((2, 2)) if need_hessian else None
    return f, g, h

def boyd_example(x: np.ndarray, need_hessian: bool = False):
    """
    f(x₁,x₂) = exp(x₁+3x₂-0.1) + exp(x₁-3x₂-0.1) + exp(-x₁-0.1)
    """
    x1, x2 = x[0], x[1]
    
    exp1 = np.exp(x1 + 3*x2 - 0.1)
    exp2 = np.exp(x1 - 3*x2 - 0.1)
    exp3 = np.exp(-x1 - 0.1)
    
    f = exp1 + exp2 + exp3
    
    g = np.array([
        exp1 + exp2 - exp3,  # df/dx₁
        3*exp1 - 3*exp2      # df/dx₂
    ])
    
    if need_hessian:
        h = np.array([
            [exp1 + exp2 + exp3,    3*exp1 - 3*exp2],
            [3*exp1 - 3*exp2,    9*exp1 + 9*exp2]
        ])
    else:
        h = None
    
    return f, g, h
