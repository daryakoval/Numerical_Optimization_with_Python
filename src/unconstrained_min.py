import numpy as np
from enum import Enum
from typing import Callable, Tuple, List, Optional

class SearchDirection(Enum):
    GRADIENT_DESCENT = "gradient_descent"
    NEWTON = "newton"

class OptimizationResult:
    """Class to store optimization results"""
    def __init__(self, iterations, gradient, x_final: np.ndarray, f_final: float, success: bool,
                 path: List[np.ndarray], objective_values: List[float], result_reason: str = ""):
        
        self.iterations = iterations
        self.gradient = gradient
        self.x_final = x_final
        self.f_final = f_final
        self.success = success
        self.path = path
        self.objective_values = objective_values
        self.result_reason = result_reason

class LineSearchOptimizer:
    
    def __init__(self, method: SearchDirection = SearchDirection.GRADIENT_DESCENT,
                 c1: float = 0.01, beta: float = 0.5):
        """        
        Args:
            method: Search direction method (gradient descent or newton)
            c1: Wolfe condition constant (default: 0.01)
            beta: Backtracking constant (default: 0.5)
        """
        self.method = method
        self.c1 = c1
        self.beta = beta

    def _compute_search_direction(self, gradient: np.ndarray, hessian: Optional[np.ndarray] = None) -> np.ndarray:
        match self.method:
            case SearchDirection.GRADIENT_DESCENT:
                return -gradient
            case SearchDirection.NEWTON:
                try:
                    return -np.linalg.solve(hessian, gradient)
                except Exception:
                    return -gradient

    def _backtracking_line_search(self, f: Callable, x: np.ndarray, direction: np.ndarray, gradient: np.ndarray, f_x: float) -> float:
        alpha = 1.0
        while True:
            x_new = x + alpha * direction
            f_new = f(x_new, False)[0]
            
            # Check Wolfe condition
            if f_new <= f_x + self.c1 * alpha * gradient.dot(direction):
                return alpha
            
            alpha *= self.beta

    def minimize(self, f: Callable, x0: np.ndarray, obj_tol: float = 1e-12,
                param_tol: float = 1e-8, max_iter: int = 100, debug=True) -> OptimizationResult:
        """
        Minimize a function using line search optimization
        
        Args:
            f: function minimized.
            x0: Starting point.
            obj_tol: numeric tolerance for successful termination due to small enough objective change or Newton Decrement.
            param_tol:  numeric tolerance for successful termination in terms of small enough distance between iterations. 
            max_iter: maximum allowed number of iterations. 
            
        Returns:
            OptimizationResult object containing optimization results
        """
        x = x0.copy()
        path = [x.copy()]
        objective_values = []
        
        # Initial function evaluation
        match self.method:
            case SearchDirection.NEWTON:
                f_x, gradient, hessian = f(x, True)
            case SearchDirection.GRADIENT_DESCENT:
                f_x, gradient = f(x, False)[:2]
                hessian = None
        
        objective_values.append(f_x)
        
        # Print initial state
        if debug:
            print(f"Iteration 0: x = {x}, f(x) = {f_x}")
        
        for i in range(max_iter):
            direction = self._compute_search_direction(gradient, hessian)
            
            match self.method:
                case SearchDirection.NEWTON:
                    newton_decrement = -gradient.dot(direction)
                    if newton_decrement/2 < obj_tol:
                        return OptimizationResult(i, gradient, x, f_x, True, path, objective_values, f"Newton decrement condition satisfied (newton_decrement/2 = {newton_decrement/2:.2e} < {obj_tol:.2e})")
                case SearchDirection.GRADIENT_DESCENT:
                    pass
            
            alpha = self._backtracking_line_search(f, x, direction, gradient, f_x)
            
            x_new = x + alpha * direction
            
            match self.method:
                case SearchDirection.NEWTON:
                    f_new, gradient_new, hessian_new = f(x_new, True)
                case SearchDirection.GRADIENT_DESCENT:
                    f_new, gradient_new = f(x_new, False)[:2]
                    hessian_new = None
            
            path.append(x_new.copy())
            objective_values.append(f_new)
            
            if debug:
                print(f"Iteration {i+1}: x = {x_new}, f(x) = {f_new}")
            
            if abs(f_new - f_x) < obj_tol:
                return OptimizationResult(i + 1, gradient, x_new, f_new, True, path, objective_values, f"Objective value convergence (|f_new - f_x| = {abs(f_new - f_x):.2e} < {obj_tol:.2e})")
            
            if np.linalg.norm(x_new - x) < param_tol:
                return OptimizationResult(i + 1, gradient, x_new, f_new, True, path, objective_values, f"Parameter convergence (||x_new - x|| = {np.linalg.norm(x_new - x):.2e} < {param_tol:.2e})")
            
            x = x_new
            f_x = f_new
            gradient = gradient_new
            hessian = hessian_new
        
        return OptimizationResult(i + 1, gradient, x, f_x, False, path, objective_values, f"Maximum iterations ({max_iter}) reached without convergence")
