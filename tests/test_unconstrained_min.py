import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.unconstrained_min import LineSearchOptimizer, SearchDirection
from src.utils import plot_contours_with_paths, plot_objective_values
from . import examples

class TestUnconstrainedMinimization(unittest.TestCase):
    def setUp(self):
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.default_max_iter = 100
        self.rosenbrock_max_iter = 10000  # Special case for Rosenbrock with GD
        
        # Initial points
        self.default_x0 = np.array([1.0, 1.0])
        self.rosenbrock_x0 = np.array([-1.0, 2.0])
        
    def print_results(self, results, title):
        for method, result in results:
            print(title)
            print(f"Final Iteration Report {method}")
            print(f"Iterations completed: {result.iterations}")
            print(f"Final x = {result.x_final}")
            print(f"Final f(x) = {result.f_final}")
            print(f"Gradient norm = {np.linalg.norm(result.gradient)}")
            print(f"Converged: {result.success}")
            print(f"Termination reason: {result.result_reason}\n\n")

    def run_optimization_test(self, f, x0, title, xlim=(-2, 2), ylim=(-2, 2),
                            max_iter_gd=None, max_iter_newton=None, debug=True):
        if max_iter_gd is None:
            max_iter_gd = self.default_max_iter
        if max_iter_newton is None:
            max_iter_newton = self.default_max_iter
            
        gd_optimizer = LineSearchOptimizer(method=SearchDirection.GRADIENT_DESCENT)
        gd_result = gd_optimizer.minimize(f, x0, self.obj_tol, self.param_tol, max_iter_gd, debug)
        
        newton_optimizer = LineSearchOptimizer(method=SearchDirection.NEWTON)
        newton_result = newton_optimizer.minimize(f, x0, self.obj_tol, self.param_tol, max_iter_newton, debug)
        
        results = [
            ("Gradient Descent", gd_result),
            ("Newton", newton_result)
        ]
        
        self.print_results(results, title)
        
        plot_contours_with_paths(f, xlim, ylim, f"{title} - Contours and Paths", results)
        plt.savefig(f"{title.lower().replace(' ', '_')}_contours.png")
        plt.close()
        
        plot_objective_values(results, f"{title} - Objective Values vs Iterations")
        plt.savefig(f"{title.lower().replace(' ', '_')}_objectives.png")
        plt.close()
        
        return gd_result, newton_result

    def test_quadratic_1(self):
        results = self.run_optimization_test(
            examples.quadratic_1,
            self.default_x0,
            "Quadratic 1 (Circular)"
        )
        for method, result in zip(["Gradient Descent", "Newton"], results):
            self.assertTrue(result.success, f"{method} failed to converge")
            self.assertLess(np.linalg.norm(result.x_final), self.param_tol,
                          f"{method} did not find the minimum at origin")

    def test_quadratic_2(self):
        gd_result, newton_result = self.run_optimization_test(
            examples.quadratic_2,
            self.default_x0,
            "Quadratic 2 (Axis-Aligned Elliptical)"
        )
        
        self.assertTrue(newton_result.success, "Newton failed to converge")
        self.assertLess(np.linalg.norm(newton_result.x_final), self.param_tol,
                    "Newton did not find the minimum at origin")
        self.assertFalse(gd_result.success, "Gradient Descent should fail to converge")

    def test_quadratic_3(self):
        gd_result, newton_result = self.run_optimization_test(
            examples.quadratic_3,
            self.default_x0,
            "Quadratic 3 (Rotated Elliptical)"
        )
            
        self.assertTrue(newton_result.success, "Newton failed to converge")
        self.assertLess(np.linalg.norm(newton_result.x_final), self.param_tol,
                    "Newton did not find the minimum at origin")
        self.assertFalse(gd_result.success, "Gradient Descent should fail to converge")

    def test_rosenbrock(self):
        gd_result, newton_result = self.run_optimization_test(
            examples.rosenbrock,
            self.rosenbrock_x0,
            "Rosenbrock Function",
            xlim=(-2, 2),
            ylim=(-1, 3),
            max_iter_gd=self.rosenbrock_max_iter
        )
        self.assertTrue(newton_result.success, "Newton failed to converge")
        self.assertTrue(gd_result.success, "Gradient Descent failed to converge")

    def test_linear(self):
        gd_result, newton_result = self.run_optimization_test(
            examples.linear,
            self.default_x0,
            "Linear Function"
        )
        self.assertFalse(newton_result.success, "Newton failed to converge")
        self.assertFalse(gd_result.success, "Gradient Descent failed to converge")

    def test_boyd_example(self):
        results = self.run_optimization_test(
            examples.boyd_example,
            self.default_x0,
            "Boyd's Example",
            xlim=(-2, 2),
            ylim=(-1, 1)
        )
        for method, result in zip(["Gradient Descent", "Newton"], results):
            self.assertTrue(result.success, f"{method} failed to converge")

if __name__ == '__main__':
    unittest.main()
