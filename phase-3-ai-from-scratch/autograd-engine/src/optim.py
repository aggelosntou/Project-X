"""
optim.py — Optimizers for the autograd engine.

All optimizers take a list of Value parameters and update their .data
field after each backward pass.

SGD: theta = theta - lr * grad
Momentum: accumulate velocity to escape local curvature
Adam: adaptive per-parameter learning rates with bias correction
"""


class SGD:
    """
    Stochastic Gradient Descent.
        theta_{t+1} = theta_t - lr * grad_t

    This is the simplest possible optimizer: move parameters in the
    direction opposite to the gradient (downhill on the loss landscape).
    """

    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class SGDMomentum:
    """
    SGD with Momentum.
        v_{t+1} = momentum * v_t + grad_t
        theta_{t+1} = theta_t - lr * v_{t+1}

    Physical intuition: the parameter acts like a ball rolling downhill.
    The velocity (v) accumulates, so the ball accelerates in consistent
    gradient directions and dampens oscillations in directions with
    alternating gradients.

    Typical momentum = 0.9: "90% of last velocity + current gradient".
    """

    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        # Initialize velocity to zero for each parameter
        self.velocity = [0.0 for _ in params]

    def step(self):
        for i, p in enumerate(self.params):
            self.velocity[i] = self.momentum * self.velocity[i] + p.grad
            p.data -= self.lr * self.velocity[i]

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class Adam:
    """
    Adaptive Moment Estimation (Adam).
    Kingma & Ba, 2014. https://arxiv.org/abs/1412.6980

    Adam maintains:
      m_t: first moment (mean of gradients) — "momentum"
      v_t: second moment (mean of squared gradients) — "variance"

    Update rule:
      m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2

    Bias correction (crucial in early training when m,v ≈ 0):
      m_hat = m_t / (1 - beta1^t)
      v_hat = v_t / (1 - beta2^t)

    Parameter update:
      theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

    Why divide by sqrt(v_hat)?
      v_hat estimates the variance of the gradient.  Parameters with
      large gradient variance (noisy) get a smaller effective step size.
      Parameters with small gradient variance (consistent) get a larger
      effective step size.  This is "adaptive learning rates per parameter".

    Why eps?
      Prevents division by zero when v_hat ≈ 0.
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # time step (for bias correction)
        self.m = [0.0 for _ in params]   # first moments
        self.v = [0.0 for _ in params]   # second moments

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2

        # Bias correction factors — these compensate for m,v being initialized at 0.
        # At t=1: correction = 1/(1-0.9) = 10 — a 10x amplification.
        # At large t: correction → 1 (no effect).
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t

        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = b1 * self.m[i] + (1 - b1) * g          # update mean
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g      # update variance

            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2

            p.data -= self.lr * m_hat / ((v_hat ** 0.5) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
