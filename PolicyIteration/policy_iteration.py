# policy_iteration.py
"""Volume 2: Policy Function Iteration.
<Name>
<Class>
<Date>
"""

import numpy as np
import scipy.linalg as la
import time
import matplotlib.pyplot as plt

# Problems 1 and 3
def value_iteration(V_0, beta=.9, N=500, W_max=1, u=np.sqrt, tol=1e-6,
                    max_iter=500):
    """Perform VI according to the Bellman optimality principle

    Parameters:
        V_0 (ndarray) - The initial guess for the value function
        beta (float) - The discount rate (between 0 and 1)
        N (int) - The number of pieces you will split the cake into
        W_max (float) - The value of W_max
        u (function) - The utility function (u(0) = 0; u' > 0; u'' < 0; and lim_{c->0+} u'(c) = \inf)
        tol (float) - The stopping criteria for the value iteration
        max_iter (int) - An alternative stopping criteria

    Returns:
        V_final (ndarray) - The discrete values for the true value function (Problem 1)
        c (list) - The amount of cake to consume at each time to maximize utility (Problem 3)
    """
    # Create discretized w
    w = np.linspace(0, W_max, N+1)
    # Create a matrix for u values 
    M = np.zeros((N+1,N+1)) + w
    M = (M.T - w).T
    M[M<0] = 0 
    U = u(M)
    for k in range(max_iter):
        # Bellman Equation
        V = np.triu((U.T + beta*V_0).T)
        V_1 = np.max(V, axis=0)
        # Policy
        pi = np.argmax(V, axis=0)
        # Stopping criteria
        if la.norm(V_1 - V_0) < tol:
            break
        # Update V
        V_0 = V_1
    return V_1, extract_policy_vector(w, pi)

# Problem 2
def extract_policy_vector(possible_W, policy_function):
    """Returns the policy vector that determines how much cake should be eaten at each time step

    Parameters:
        possible_W (ndarray) - an array representing the discrete values of W
        policy_function (ndarray) - an array representing how many pieces to leave at each state

    Returns:
        c (list) - a list representing how much cake to eat at each time period
    """
    n = len(possible_W)-1
    W_max = possible_W[-1]
    w1 = possible_W[1]
    c = []
    w = sum(c)
    # Stopping criteria
    while not np.isclose(w, W_max):
        wi = W_max - w
        eat = policy_function[int(round(wi*n))]
        # Keep track of how much cake is eaten each iteration
        c.append(wi - w1*eat)
        w = sum(c)
    return c

# Problem 4
def policy_iteration(pi_0, beta=.9, N=500, W_max=1, u=np.sqrt, tol=1e-6, max_iter=50):
    """Perform PI according to the Bellman optimality principle

    Parameters:
        pi_0 (array) - The initial guess for the Policy Function (0 <= pi_0(W) <= W)
        beta (float) - The discount rate (between 0 and 1)
        N (int) - The number of pieces you will split the cake into
            also acts as a cap for the number of steps required to calculate V_k at each iteration
        W_max (float) - The value of W_max
        u (function) - The utility function (u(0) = 0; u' > 0; u'' < 0; and lim_{c->0+} u'(c) = \inf)
        max_iter (int) - An alternative stopping criteria for the policy function updates

    Returns:
        V_final (ndarray) - The discrete values for the true value function
        c (list) - The amount of cake to consume at each time to maximize utility
    """
    # Discretize w
    w = np.linspace(0, W_max, N+1)
    # Create a matrix of u values
    M = np.zeros((N+1,N+1)) + w
    M = (M.T - w).T
    M[M<0] = 0 
    U = u(M.T)
    for k in range(max_iter):
        V = np.zeros(N+1)
        # 19.7
        for i in range(1, len(w)):
            p = w[pi_0[np.where(w==w[i])[0][0]]]
            V[i] = U[np.where(w==w[i])[0][0],np.where(w==p)[0][0]] + beta*V[np.where(w==p)]
        Up = np.copy(U)
        Up[:] += beta*V[:]
        Up = np.tril(Up)
        # 19.5
        pi_1 = Up.argmax(axis=1)
        # Stopping criteria
        if la.norm(pi_1 - pi_0) < tol:
            break
        # Update pi
        pi_0 = pi_1
    return V, extract_policy_vector(w, pi_1)
# Problem 5
def compare_methods():
    """
    Solve the cake eating problem with each method, VI, PI with various values of beta and compare how long each method takes.
    Each V_final should be np.allclose and each policy vector, c, should be identical for each both.
    Use N=1000 as the number of grid points for w and beta = [.95, .975, .99, .995].

    Graph the results for each method with beta on the x-axis and time on the y-axis.
    """
    N = 1000
    betas = [.95, .975, .99, .995]
    # Initial conditions
    V0 = np.sqrt(np.linspace(0,1,1001))
    pi_0 = np.arange(-1,1000)
    pi_0[0] = 0
    # Save the times in a list
    vi_time = []
    pi_time = []
    for b in betas:
        t1 = time.time()
        value_iteration(V0, b, N)
        t2 = time.time()
        policy_iteration(pi_0, b, N)
        t3 = time.time()
        vi_time.append(t2-t1)
        pi_time.append(t3-t2)
    # Plot the result
    plt.plot(betas, vi_time, label="VI Time")
    plt.plot(betas, pi_time, label="PI Time")
    plt.xlabel("Beta Values")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.show()

V_0 = np.sqrt(np.linspace(0,1,401))
w = np.linspace(0,1,400)
pi_0 = np.arange(-1, 400)
pi_0[0] = 0
w = [0,0,1,2,2,3,3,4,4,5,5]
print(value_iteration(V_0, .995, 400))
print(extract_policy_vector(np.linspace(0,1,11), w))
print(policy_iteration(pi_0, .995, 400))
compare_methods()