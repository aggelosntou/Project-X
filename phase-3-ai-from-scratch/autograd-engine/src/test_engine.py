"""
test_engine.py — Verify autograd gradients via analytical derivation and PyTorch comparison.

Two verification strategies:
1. ANALYTICAL: For simple expressions, compute the expected gradient by hand
   and compare against our engine's output.
2. PYTORCH: If PyTorch is installed, build the same computation graph in torch
   and verify our gradients match (to float32 precision).

Why verify gradients? A silent bug in autograd (wrong chain rule in one op) will
cause training to fail with no obvious error message — gradients will just be wrong
and loss won't decrease.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import math
from engine import Value

# ------------------------------------------------------------------ #
#  Test harness                                                        #
# ------------------------------------------------------------------ #

tests_run = 0
tests_passed = 0

def check(got, expected, tol=1e-5, msg=""):
    global tests_run, tests_passed
    tests_run += 1
    diff = abs(got - expected)
    if diff <= tol:
        tests_passed += 1
        print(f"  PASS: {msg}  (got={got:.6f}, expected={expected:.6f})")
    else:
        print(f"  FAIL: {msg}  (got={got:.6f}, expected={expected:.6f}, diff={diff:.2e})")

# ------------------------------------------------------------------ #
#  Analytical tests                                                    #
# ------------------------------------------------------------------ #

def test_simple_add_mul():
    """
    L = (a + b) * c  where a=2, b=3, c=4
    L = 5 * 4 = 20
    dL/da = c = 4
    dL/db = c = 4
    dL/dc = a + b = 5
    """
    print("\n--- test_simple_add_mul ---")
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)
    L = (a + b) * c
    L.backward()
    check(a.grad, 4.0, msg="dL/da = c = 4")
    check(b.grad, 4.0, msg="dL/db = c = 4")
    check(c.grad, 5.0, msg="dL/dc = a+b = 5")

def test_pow():
    """
    L = x^3  at x=2
    dL/dx = 3 * x^2 = 12
    """
    print("\n--- test_pow ---")
    x = Value(2.0)
    L = x ** 3
    L.backward()
    check(x.grad, 12.0, msg="d(x^3)/dx at x=2 = 12")

def test_chain_rule():
    """
    L = (x^2 + y)^2  at x=1, y=2
    Let u = x^2 + y = 3
    L = u^2 = 9
    dL/du = 2u = 6
    dL/dx = dL/du * du/dx = 6 * 2x = 6 * 2 = 12
    dL/dy = dL/du * du/dy = 6 * 1 = 6
    """
    print("\n--- test_chain_rule ---")
    x = Value(1.0)
    y = Value(2.0)
    u = x**2 + y
    L = u**2
    L.backward()
    check(x.grad, 12.0, msg="d((x^2+y)^2)/dx at x=1,y=2 = 12")
    check(y.grad,  6.0, msg="d((x^2+y)^2)/dy at x=1,y=2 = 6")

def test_tanh():
    """
    L = tanh(x)  at x=0.5
    dL/dx = 1 - tanh^2(x) = sech^2(x)
    """
    print("\n--- test_tanh ---")
    x = Value(0.5)
    L = x.tanh()
    L.backward()
    expected = 1 - math.tanh(0.5)**2
    check(x.grad, expected, tol=1e-6, msg="d(tanh(x))/dx at x=0.5")

def test_sigmoid():
    """
    L = sigmoid(x)  at x=1.0
    s = 1/(1+e^{-1})
    dL/dx = s*(1-s)
    """
    print("\n--- test_sigmoid ---")
    x = Value(1.0)
    L = x.sigmoid()
    L.backward()
    s = 1 / (1 + math.exp(-1.0))
    expected = s * (1 - s)
    check(x.grad, expected, tol=1e-6, msg="d(sigmoid(x))/dx at x=1")

def test_exp_log():
    """
    L = log(exp(x)) = x  (identity)
    dL/dx should be 1.0
    """
    print("\n--- test_exp_log ---")
    x = Value(1.5)
    L = x.exp().log()
    L.backward()
    check(x.grad, 1.0, tol=1e-5, msg="d(log(exp(x)))/dx = 1")

def test_multiuse_node():
    """
    When a node is used multiple times, its gradient should accumulate.
    L = x * x = x^2  at x=3
    dL/dx = 2x = 6

    The gradient accumulates: from the left multiply (dL/dx += x = 3)
    and from the right multiply (dL/dx += x = 3), total = 6.
    """
    print("\n--- test_multiuse_node ---")
    x = Value(3.0)
    L = x * x
    L.backward()
    check(x.grad, 6.0, msg="d(x*x)/dx at x=3 = 6 (accumulation)")

def test_relu():
    """
    L = relu(x)
    dL/dx = 1 if x > 0, else 0
    """
    print("\n--- test_relu ---")
    x1 = Value(2.0)
    L1 = x1.relu()
    L1.backward()
    check(x1.grad, 1.0, msg="relu backward at x=2 (active)")

    x2 = Value(-1.0)
    L2 = x2.relu()
    L2.backward()
    check(x2.grad, 0.0, msg="relu backward at x=-1 (dead)")

def test_division():
    """
    L = a / b  at a=6, b=2
    L = 3
    dL/da = 1/b = 0.5
    dL/db = -a/b^2 = -6/4 = -1.5
    """
    print("\n--- test_division ---")
    a = Value(6.0)
    b = Value(2.0)
    L = a / b
    L.backward()
    check(a.grad,  0.5, msg="d(a/b)/da at a=6,b=2 = 0.5")
    check(b.grad, -1.5, msg="d(a/b)/db at a=6,b=2 = -1.5")

# ------------------------------------------------------------------ #
#  PyTorch comparison (optional)                                       #
# ------------------------------------------------------------------ #

def test_vs_pytorch():
    """
    Build the same computation graph in PyTorch and verify gradients match.
    """
    print("\n--- test_vs_pytorch ---")
    try:
        import torch

        # Expression: L = ((a * b + c) ** 2).tanh()
        a_val, b_val, c_val = 1.5, 2.0, -0.5

        # Our engine
        a = Value(a_val)
        b = Value(b_val)
        c = Value(c_val)
        L_ours = ((a * b + c) ** 2).tanh()
        L_ours.backward()

        # PyTorch
        a_t = torch.tensor(a_val, requires_grad=True, dtype=torch.float64)
        b_t = torch.tensor(b_val, requires_grad=True, dtype=torch.float64)
        c_t = torch.tensor(c_val, requires_grad=True, dtype=torch.float64)
        L_pt = ((a_t * b_t + c_t) ** 2).tanh()
        L_pt.backward()

        check(a.grad, float(a_t.grad), tol=1e-5, msg="dL/da matches PyTorch")
        check(b.grad, float(b_t.grad), tol=1e-5, msg="dL/db matches PyTorch")
        check(c.grad, float(c_t.grad), tol=1e-5, msg="dL/dc matches PyTorch")

    except ImportError:
        print("  SKIP: PyTorch not installed")

# ------------------------------------------------------------------ #
#  Run all tests                                                       #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    print("=== Autograd Engine Tests ===")
    test_simple_add_mul()
    test_pow()
    test_chain_rule()
    test_tanh()
    test_sigmoid()
    test_exp_log()
    test_multiuse_node()
    test_relu()
    test_division()
    test_vs_pytorch()

    print(f"\n=== Results: {tests_passed} / {tests_run} passed ===")
    sys.exit(0 if tests_passed == tests_run else 1)
