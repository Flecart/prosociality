"""Numerical check: in linear-quadratic games the HOLA (higher-order LOLA)
look-ahead recursion is EXACTLY a Neumann series, isomorphic to Bergstrom's
interdependence fixed point U = pi + A U  =>  U = (I-A)^{-1} pi.

HOLA recursion (Willi et al. 2022, COLA), exact form:
    h1^{n+1} = -alpha * grad_1 L1(t1, t2 + h2^n)
    h2^{n+1} = -alpha * grad_2 L2(t1 + h1^n, t2)
Consistent (iLOLA) fixed point:
    f1 = -alpha grad_1 L1(t1, t2 + f2),  f2 = -alpha grad_2 L2(t1 + f1, t2)

For quadratic L^i the gradients are affine, so
    grad_1 L1(t1, t2 + h2) = grad_1 L1(t1,t2) + H1_12 h2
with H1_12 the cross-Hessian d/dt2 (grad_1 L1). Hence
    h^{n+1} = g + M h^n,   M = [[0, -alpha H1_12],[-alpha H2_21, 0]],  g = naive update.
This is U = pi + A U with (U,pi,A) <-> (h, g, M).  Partial sum HOLA_n = sum_{k<=n} M^k g;
fixed point = (I-M)^{-1} g; HOLA converges iff rho(M) < 1.
"""
import numpy as np

rng = np.random.default_rng(0)


def quad_game(scale):
    """Random 2-player scalar-parameter quadratic game; returns gradient fns and Hessian blocks.

    L^i(t1,t2) = 0.5 a_i t_i^2 + c_i t1 t2 + b_i t_i  (own curvature a_i>0, coupling c_i).
    grad_i L^i = a_i t_i + c_i t_{-i} + b_i.   cross-Hessian d(grad_i L^i)/dt_{-i} = c_i.
    """
    a1, a2 = 1.0, 1.0
    c1 = scale * rng.normal()
    c2 = scale * rng.normal()
    b1, b2 = rng.normal(), rng.normal()

    def grad1(t1, t2):
        return a1 * t1 + c1 * t2 + b1

    def grad2(t1, t2):
        return a2 * t2 + c2 * t1 + b2

    return grad1, grad2, c1, c2


def run(scale, alpha, n_iters=200):
    grad1, grad2, c1, c2 = quad_game(scale)
    t1, t2 = 0.3, -0.4  # arbitrary evaluation point

    # naive updates g_i = -alpha grad_i L^i  (HOLA^0)
    g = np.array([-alpha * grad1(t1, t2), -alpha * grad2(t1, t2)])
    # coupling matrix M: block-antidiagonal scaled cross-Hessians
    M = np.array([[0.0, -alpha * c1], [-alpha * c2, 0.0]])

    # --- exact HOLA recursion (differentiate through opponent's anticipated step) ---
    h = np.zeros(2)
    for _ in range(n_iters):
        h_new = np.array([
            -alpha * grad1(t1, t2 + h[1]),
            -alpha * grad2(t1 + h[0], t2)
        ])
        h = h_new
    hola_limit = h.copy()

    # --- Neumann partial sum and closed-form inverse ---
    neumann = np.zeros(2)
    Mk = np.eye(2)
    for _ in range(n_iters):
        neumann = neumann + Mk @ g
        Mk = Mk @ M
    rho = max(abs(np.linalg.eigvals(M)))
    inv = np.linalg.solve(np.eye(2) - M, g) if abs(np.linalg.det(np.eye(2) - M)) > 1e-9 else None

    return rho, g, M, hola_limit, neumann, inv


print(f"{'scale':>6} {'alpha':>6} {'rho(M)':>8}  {'HOLA conv?':>10}  match (I-M)^-1 g?")
for scale, alpha in [(0.3, 0.5), (0.7, 0.5), (0.9, 1.0), (1.3, 1.0), (2.2, 1.0), (3.0, 1.2)]:
    rho, g, M, hola, neumann, inv = run(scale, alpha)
    converged = np.all(np.isfinite(hola)) and np.linalg.norm(hola) < 1e6
    if inv is not None and converged:
        match = np.allclose(hola, inv, atol=1e-6) and np.allclose(neumann, inv, atol=1e-6)
        msg = f"YES  err={np.linalg.norm(hola-inv):.2e}" if match else "NO"
    else:
        msg = "HOLA diverged; (I-M)^-1 still defined" if inv is not None else "singular"
    print(f"{scale:6.2f} {alpha:6.2f} {rho:8.4f}  {str(converged):>10}  {msg}")

print("\nInterpretation:")
print(" rho(M)<1 -> HOLA (Neumann series) converges to the consistent look-ahead = (I-M)^-1 g.")
print(" rho(M)>=1 -> HOLA diverges, yet (I-M)^-1 g (the COLA consistent fixed point) still exists.")
print(" Exactly Bergstrom: U=pi+AU, U=(I-A)^-1 pi, Neumann sum_k A^k converges iff rho(A)<1.")
