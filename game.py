# Python code to model three Stackelberg formulations for Sustainable Aviation Fuel (SAF)
# 1) Single government (leader) — single airline (follower)
# 2) Single government (leader) — N symmetric airlines (followers; Cournot among followers)
# 3) Two governments (simultaneous leaders) — two groups of airlines (followers) with cross-market price
#
# The model assumptions (simple, transparent):
# - Inverse market price for SAF: P(Q_total) = a - b * Q_total
# - Each airline i gains a per-unit "value" v from using SAF (reputation / avoided penalty)
# - Each airline receives a per-unit subsidy s (set by its government) applied to its purchases
# - Airline i's payoff: U_i = v * Q_i - (P(Q_total) - s_home) * Q_i  (choose Q_i >= 0)
# - Government j's welfare: W_j = beta * Q_total - s_j * Q_j_total
#   (beta quantifies the social value of SAF uptake; subsidy cost is s_j times quantity bought by its airlines)
#
# The code provides:
# - Analytic formulas where possible (followers' symmetric closed-form)
# - Numerical optimization for leader(s) using scipy (with fallback grid search)
# - Example runs for chosen parameters
#
# NOTE: This is a stylized model to illustrate how multi-follower and multi-leader setups change equilibria.
# You can adapt payoff functions, costs, or objectives to better match your SAF context.

import numpy as np
from scipy.optimize import minimize_scalar

# ------------------------- Utility functions -------------------------

def price_inverse(Q_total, a, b):
    """Inverse demand / market price: P(Q) = a - b Q_total"""
    return np.maximum(a - b * Q_total, 0.0)

# --------------------- 1) Single government, single airline ---------------------
def follower_quantity_single(a, b, v, s):
    """
    Analytic best-response for a single airline (follower) when government subsidy = s.
    Follower maximizes U = v Q - (P(Q) - s) Q with P(Q) = a - b Q.
    First-order condition leads to Q = (a - s - v) / (2 b). Enforce non-negativity.
    """
    Q = (a - s - v) / (2.0 * b)
    return max(0.0, Q)

def government_welfare_single(a, b, v, beta, s):
    """
    Given subsidy s, compute follower quantity and government welfare:
    W = beta * Q - s * Q
    """
    Q = follower_quantity_single(a, b, v, s)
    return (beta - s) * Q, Q  # returns (welfare, quantity)

def solve_single_gov_single_airline(a=10.0, b=1.0, v=1.0, beta=5.0, s_bounds=(0,9)):
    """
    Solve for optimal subsidy s* for the single government (leader) anticipating single follower response.
    """
    # maximize welfare -> minimize negative welfare
    res = minimize_scalar(lambda s: -government_welfare_single(a,b,v,beta,s)[0],
                              bounds=s_bounds, method='bounded')
    s_star = float(res.x)
    W_star, Q_star = government_welfare_single(a, b, v, beta, s_star)
    return {'s_star': s_star, 'Q_star': Q_star, 'W_star': W_star}


# --------------------- 2) Single government, N symmetric airlines ---------------------
def symmetric_cournot_quantities(a, b, v, s, N):
    """
    Closed-form symmetric Nash equilibrium for N identical airlines under subsidy s:
    q_i = (a - s - v) / (b*(N + 1))
    Enforce non-negativity.
    """
    numerator = a - s - v
    if numerator <= 0:
        q = 0.0
    else:
        q = numerator / (b * (N + 1.0))
    Q_total = N * q
    return q, Q_total

def government_welfare_multi(a, b, v, beta, s, N):
    q, Q_total = symmetric_cournot_quantities(a,b,v,s,N)
    # Government welfare: beta * Q_total - subsidy_cost (s * Q_total)
    W = (beta - s) * Q_total
    return W, q, Q_total

def solve_single_gov_multi_airlines(a=10.0, b=1.0, v=1.0, beta=5.0, N=3, s_bounds=(0,9)):
    """
    Government chooses subsidy s anticipating N symmetric airlines' Nash equilibrium.
    """
    res = minimize_scalar(lambda s: -government_welfare_multi(a,b,v,beta,s,N)[0],
                              bounds=s_bounds, method='bounded')
    s_star = float(res.x)
    W_star, q_star, Q_star = government_welfare_multi(a,b,v,beta,s_star,N)
    return {'s_star': s_star, 'q_per_airline': q_star, 'Q_total': Q_star, 'W_star': W_star}


# --------------------- 3) Two governments (leaders) — two groups of airlines (followers) ---------------------
def symmetric_two_group_quantities(a, b, v, s1, s2, N1, N2):
    """
    Solve symmetric equilibrium per-airline quantities q1 and q2 when:
    - Group 1 (N1 airlines) get subsidy s1
    - Group 2 (N2 airlines) get subsidy s2
    Derivation (linear system):
      (N1+1) q1 + N2 q2 = (a - s1 - v) / b
      N1 q1 + (N2+1) q2 = (a - s2 - v) / b
    Solve for q1 and q2, enforce non-negativity.
    """
    rhs1 = (a - s1 - v) / b
    rhs2 = (a - s2 - v) / b
    A = np.array([[N1 + 1.0, N2],
                  [N1,       N2 + 1.0]])
    rhs = np.array([rhs1, rhs2])
    try:
        sol = np.linalg.solve(A, rhs)
        q1, q2 = sol[0], sol[1]
    except np.linalg.LinAlgError:
        # fallback - singular matrix (degenerate), set zeros
        q1, q2 = 0.0, 0.0
    # enforce non-negativity (if negative -> set to zero and recompute numerically by clamping)
    q1 = max(0.0, q1)
    q2 = max(0.0, q2)
    Q_total = N1 * q1 + N2 * q2
    return q1, q2, Q_total

def governments_welfare_two(a, b, v, beta, s1, s2, N1, N2):
    q1, q2, Q_total = symmetric_two_group_quantities(a,b,v,s1,s2,N1,N2)
    # Government 1 welfare
    W1 = (beta - s1) * (N1 * q1)
    W2 = (beta - s2) * (N2 * q2)
    return W1, W2, q1, q2, Q_total

def solve_two_govs_two_groups(a=10.0, b=1.0, v=1.0, beta=5.0, N1=2, N2=3, s_bounds=(0,9), max_iter=200, tol=1e-6):
    """
    Solve for a Nash equilibrium between two simultaneous leaders (governments) that set subsidies s1 and s2.
    Each government best-responds to the other's subsidy by maximizing its own welfare,
    where the follower equilibrium (quantities q1,q2) is computed analytically for given (s1,s2).
    We'll compute the leaders' Nash equilibrium via iterative best-response (Gauss-Seidel style).
    """
    # Initialization
    s1 = 0.5 * (s_bounds[0] + s_bounds[1])
    s2 = 0.5 * (s_bounds[0] + s_bounds[1])
    # We'll compute exact best-response using scalar optimization for each leader
    for it in range(max_iter):
        # Government 1 best response given s2:
        res1 = minimize_scalar(lambda s: -governments_welfare_two(a,b,v,beta,s,s2,N1,N2)[0],
                                   bounds=s_bounds, method='bounded')
        s1_new = float(res1.x)
        # Government 2 best response given s1_new:
        res2 = minimize_scalar(lambda s: -governments_welfare_two(a,b,v,beta,s1_new,s,N1,N2)[1],
                                   bounds=s_bounds, method='bounded')
        s2_new = float(res2.x)
        if abs(s1_new - s1) < tol and abs(s2_new - s2) < tol:
            s1, s2 = s1_new, s2_new
            break
        s1, s2 = s1_new, s2_new

    # compute final quantities/welfares
    W1, W2, q1, q2, Q_total = governments_welfare_two(a,b,v,beta,s1,s2,N1,N2)
    return {'s1_star': s1, 's2_star': s2, 'q1_per_airline': q1, 'q2_per_airline': q2,
            'Q_total': Q_total, 'W1': W1, 'W2': W2}

# --------------------- Example runs ---------------------
if __name__ == "__main__":
    # Model parameters (example)
    params = {
        'a': 12.0,   # demand intercept (how high price can be if Q=0)
        'b': 1.0,    # demand slope
        'v': 1.5,    # per-unit value to airlines for using SAF (could represent avoided penalties)
        'beta': 6.0  # social value per unit SAF for government
    }

    print("\n--- 1) Single Government, Single Airline ---")
    sol1 = solve_single_gov_single_airline(**params, s_bounds=(0, params['a'] - params['v'] - 1e-6))
    print(f"Optimal subsidy s* = {sol1['s_star']:.4f}, Airline Q* = {sol1['Q_star']:.4f}, Government W = {sol1['W_star']:.4f}")

    print("\n--- 2) Single Government, N symmetric Airlines ---")
    sol2 = solve_single_gov_multi_airlines(**params, N=4, s_bounds=(0, params['a'] - params['v'] - 1e-6))
    print(f"Optimal subsidy s* = {sol2['s_star']:.4f}, Per-airline q = {sol2['q_per_airline']:.4f}, Q_total = {sol2['Q_total']:.4f}, Government W = {sol2['W_star']:.4f}")

    print("\n--- 3) Two Governments, Two Groups of Airlines ---")
    sol3 = solve_two_govs_two_groups(**params, N1=2, N2=3, s_bounds=(0, params['a'] - params['v'] - 1e-6))
    print(f"Gov1 s* = {sol3['s1_star']:.4f}, gov2 s* = {sol3['s2_star']:.4f}")
    print(f"Per-airline q1 = {sol3['q1_per_airline']:.4f}, q2 = {sol3['q2_per_airline']:.4f}, Q_total = {sol3['Q_total']:.4f}")
    print(f"Gov1 welfare = {sol3['W1']:.4f}, Gov2 welfare = {sol3['W2']:.4f}")

# End of script. You can adapt the utility/welfare functions (e.g., add costs, supply-side pricing, emissions penalties)
# to better reflect your SAF supply-chain specifics.
