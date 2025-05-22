import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple
from .unconstrained_min import OptimizationResult

def plot_contours_with_paths(f: Callable, xlim: Tuple[float, float], 
                           ylim: Tuple[float, float], title: str,
                           results: List[Tuple[str, OptimizationResult]] = None,
                           levels: int = 70):
    """
    Plot contour lines of the objective function and optimization paths
    
    Args:
        f: Objective function that takes (x, need_hessian) as input
        xlim: Tuple of (min_x, max_x) for plot limits
        ylim: Tuple of (min_y, max_y) for plot limits
        title: Plot title
        results: List of (method_name, OptimizationResult) tuples to plot paths
        levels: Number of contour levels
    """
    # Create grid points
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate function on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = f(np.array([X[i,j], Y[i,j]]), False)[0]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    min_z, max_z = Z.min(), Z.max()
    if max_z - min_z < 1e-8 or min_z <= 0:
        # Fall back to linear spacing if logspace is not valid
        contour_levels = np.linspace(min_z, max_z + 1e-3, levels)
    else:
        contour_levels = np.logspace(np.log10(min_z + 1e-6), np.log10(max_z), levels)

    cs = plt.contour(X, Y, Z, levels=contour_levels)
    plt.colorbar(cs, label='f(x)')
    
    # Plot optimization paths if provided
    if results is not None:
        for method_name, result in results:
            path = np.array(result.path)
            if 'Gradient' in method_name:
                plt.plot(path[:,0], path[:,1], 'o-', color='orange', label=method_name, linewidth=2,
                        markersize=4, markerfacecolor='orange')
                plt.plot(path[0,0], path[0,1], 'g*', markersize=12, label=f'{method_name} start')
                plt.plot(path[-1,0], path[-1,1], 'r*', markersize=12, label=f'{method_name} end')
            else:  # Newton's method
                plt.plot(path[:,0], path[:,1], 'D:', color='blue', label=method_name, linewidth=2, 
                        markersize=4, alpha=0.7, markerfacecolor='white', markeredgecolor='blue')
                plt.plot(path[0,0], path[0,1], 'g*', markersize=12, label=f'{method_name} start')
                plt.plot(path[-1,0], path[-1,1], 'r*', markersize=12, label=f'{method_name} end')
    
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.axis('equal')

def plot_objective_values(results: List[Tuple[str, OptimizationResult]], title: str):
    """
    Plot objective function values vs iteration number for different methods
    
    Args:
        results: List of (method_name, OptimizationResult) tuples to compare
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    for method_name, result in results:
        iterations = range(len(result.objective_values))
        if 'Gradient' in method_name:
            plt.semilogy(iterations, result.objective_values, 'o-', color='orange', 
                        label=method_name, linewidth=5, markersize=8, 
                        alpha=0.7, markerfacecolor='orange', markeredgecolor='orange')
        else:  # Newton's method
            plt.semilogy(iterations, result.objective_values, 'o-', color='blue',
                        label=method_name, linewidth=2, markersize=6,
                        markerfacecolor='blue')
    
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value (log scale)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
