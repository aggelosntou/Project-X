"""
test_functions.py — Standard optimization benchmark functions with exact gradients.

These are classic functions used to evaluate optimization algorithms.
Each has known global minima and varying difficulty characteristics:
  - Rosenbrock: narrow curved valley, gradient alignment changes along path
  - Beale:      flat regions with steep walls
  - Himmelblau: multiple global optima
  - Rastrigin:  many local minima (highly multimodal)

All functions take (x, y) scalars and return (f_value, gradient_array).
"""

import numpy as np


def rosenbrock(x: float, y: float) -> tuple:
    """
    Rosenbrock function (Rosenbrock, 1960).

    f(x, y) = (1 - x)² + 100(y - x²)²

    Global minimum: f(1, 1) = 0

    Properties:
    - The minimum lies inside a narrow, curved, parabolic valley
    - Easy to find the valley, hard to converge to the minimum
    - Gradient points away from the valley axis — optimizers must follow a
      curved path, not a straight line
    - Tests second-order information sensitivity (momentum helps a lot)
    - Condition number of Hessian at minimum: ~2500 for the standard version

    Gradient:
      df/dx = -2(1-x) - 400x(y-x²)
      df/dy = 200(y-x²)
    """
    f  = (1 - x)**2 + 100 * (y - x**2)**2
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return f, np.array([dx, dy])


def beale(x: float, y: float) -> tuple:
    """
    Beale function.

    f(x, y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²

    Global minimum: f(3, 0.5) = 0
    Domain: -4.5 <= x, y <= 4.5

    Properties:
    - Large flat regions with a steep, narrow valley toward the minimum
    - Gradient nearly zero far from minimum → SGD stalls, adaptive methods help
    - Tests sensitivity to flat plateaus

    Gradient (by chain rule):
      Let r1 = 1.5 - x + xy,   r2 = 2.25 - x + xy²,   r3 = 2.625 - x + xy³
      df/dx = 2*r1*(-1+y) + 2*r2*(-1+y²) + 2*r3*(-1+y³)
      df/dy = 2*r1*(x) + 2*r2*(2xy) + 2*r3*(3xy²)
    """
    r1 = 1.5   - x + x * y
    r2 = 2.25  - x + x * y**2
    r3 = 2.625 - x + x * y**3

    f = r1**2 + r2**2 + r3**2

    dx = (2 * r1 * (-1 + y) +
          2 * r2 * (-1 + y**2) +
          2 * r3 * (-1 + y**3))
    dy = (2 * r1 * x +
          2 * r2 * (2 * x * y) +
          2 * r3 * (3 * x * y**2))

    return f, np.array([dx, dy])


def himmelblau(x: float, y: float) -> tuple:
    """
    Himmelblau's function (Himmelblau, 1972).

    f(x, y) = (x² + y - 11)² + (x + y² - 7)²

    Four global minima (all with f = 0):
      (3.0,    2.0)
      (-2.805, 3.131)
      (-3.779, -3.283)
      (3.584,  -1.848)

    Properties:
    - Multiple global optima of equal value
    - Tests robustness: which minimum does the optimizer find?
    - Convergence depends strongly on initialization
    - One saddle point and one local maximum

    Gradient:
      df/dx = 4x(x² + y - 11) + 2(x + y² - 7)
      df/dy = 2(x² + y - 11) + 4y(x + y² - 7)
    """
    r1 = x**2 + y - 11
    r2 = x + y**2 - 7

    f  = r1**2 + r2**2
    dx = 4 * x * r1 + 2 * r2
    dy = 2 * r1 + 4 * y * r2

    return f, np.array([dx, dy])


def rastrigin(x: float, y: float) -> tuple:
    """
    Rastrigin function (Rastrigin, 1974).

    f(x, y) = 20 + x² - 10cos(2πx) + y² - 10cos(2πy)

    Global minimum: f(0, 0) = 0
    Domain: typically -5.12 <= x, y <= 5.12

    Properties:
    - Highly multimodal: many local minima on a lattice at integer points
    - The global minimum is at (0, 0) but surrounded by many local minima
    - Tests ability to escape local minima
    - Gradient-based methods almost always find a local minimum nearby
    - Standard benchmark showing limitations of gradient-only optimization

    Number of local minima: approximately (10/π)² ≈ 10.1 per unit area

    Gradient:
      df/dx = 2x + 20π sin(2πx)
      df/dy = 2y + 20π sin(2πy)
    """
    f  = 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)
    dx = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    dy = 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)

    return f, np.array([dx, dy])


def numerical_gradient(fn, x: float, y: float, eps: float = 1e-5) -> np.ndarray:
    """
    Compute gradient via central finite differences (for verification).

    df/dx ≈ (f(x+ε, y) - f(x-ε, y)) / (2ε)
    df/dy ≈ (f(x, y+ε) - f(x, y-ε)) / (2ε)
    """
    f_xp, _ = fn(x + eps, y)
    f_xm, _ = fn(x - eps, y)
    f_yp, _ = fn(x, y + eps)
    f_ym, _ = fn(x, y - eps)

    dx = (f_xp - f_xm) / (2 * eps)
    dy = (f_yp - f_ym) / (2 * eps)
    return np.array([dx, dy])


def verify_gradients():
    """Verify all analytical gradients against numerical finite differences."""
    test_points = [
        (0.5, 0.5), (-1.0, 1.0), (2.0, -0.5), (0.0, 0.0)
    ]
    functions = [
        ('Rosenbrock', rosenbrock),
        ('Beale',      beale),
        ('Himmelblau', himmelblau),
        ('Rastrigin',  rastrigin),
    ]

    print("Gradient verification (analytical vs numerical):")
    all_passed = True
    for fname, fn in functions:
        for x, y in test_points:
            _, ana_grad = fn(x, y)
            num_grad    = numerical_gradient(fn, x, y)
            err = np.abs(ana_grad - num_grad).max()
            status = "PASS" if err < 1e-5 else "FAIL"
            if status == "FAIL":
                all_passed = False
                print(f"  {fname} at ({x},{y}): {status} (err={err:.2e})")

    if all_passed:
        print("  All gradient checks PASSED.")
    return all_passed


if __name__ == "__main__":
    print("=== Test Function Verification ===\n")

    # Print function values at known optima
    print("Function values at known optima:")
    f, g = rosenbrock(1.0, 1.0)
    print(f"  Rosenbrock(1,1)         = {f:.6f}  (expected 0.0)")

    f, g = beale(3.0, 0.5)
    print(f"  Beale(3, 0.5)           = {f:.6f}  (expected 0.0)")

    f, g = himmelblau(3.0, 2.0)
    print(f"  Himmelblau(3, 2)        = {f:.6f}  (expected 0.0)")

    f, g = himmelblau(-2.805118, 3.131312)
    print(f"  Himmelblau(-2.8, 3.1)   = {f:.6f}  (expected ~0.0)")

    f, g = rastrigin(0.0, 0.0)
    print(f"  Rastrigin(0, 0)         = {f:.6f}  (expected 0.0)")

    print()
    verify_gradients()

    # Show gradient norms at starting point (-1, 1) for Rosenbrock
    print("\nGradient at Rosenbrock starting point (-1, 1):")
    f, g = rosenbrock(-1.0, 1.0)
    print(f"  f(-1, 1) = {f:.2f},  grad = {g},  ||grad|| = {np.linalg.norm(g):.2f}")
