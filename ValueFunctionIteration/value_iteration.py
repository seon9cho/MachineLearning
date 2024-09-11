# value_iteration.py
"""Volume 2: Value Function Iteration.
<Name>
<Class>
<Date>
"""

import numpy as np
import matplotlib.pyplot as plt

# Problem 1
def graph_policy(policy, b, u):
    """Plot the utility gained over time.
    Return the total utility gained with the policy given.

    Parameters:
        policy (ndarray): Policy vector.
        b (float): Discount factor. 0 < beta < 1.
        u (function): Utility function.

    Returns:
        total_utility (float): Total utility gained from the policy given.
    """
    value = np.array([b**t*u(c) for t,c in enumerate(policy)])
    value_sum = np.array([sum(value[:i+1]) for i in range(len(policy))])
    domain = np.linspace(0, len(policy)-1, len(policy))
    print(domain)
    plt.plot(domain, value_sum, label="Policy, Utility="+str(b))
    plt.legend()
    plt.show()
    return value_sum

# Problem 2
def consumption(N, u=np.sqrt):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): Consumption matrix.
    """
    W = np.linspace(0, 1, N+1)
    M = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            if W[i]-W[j] >= 0:
                M[i,j] += u(W[i]-W[j])
    return M
    
# Problems 3-5
def eat_cake(T, N, B, u=np.sqrt):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    A = np.zeros((N+1, T+1))
    A[:,-1] = np.array([u(i/N) for i in range(N+1)])
    P = np.zeros((N+1,T+1))
    P[:,-1] = np.array([i/N for i in range(N+1)])
    for i in range(T):
        CV = consumption(N, u=u)
        for j in range(N+1):
            CV[:,j] += b*A[j,-(1+i)]
            CV[:,j][:j] = 0
        P[:,-(2+i)] = P[:,-1] - P[:,-1][np.argmax(CV, axis=1)]
        A[:,-(2+i)] = np.max(CV, axis=1)        
    return A, P


# Problem 6
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        maximum_utility (float): The total utility gained from the
            optimal policy.
        optimal_policy ((N,) nd array): The matrix describing the optimal
            percentage to consume at each time.
    """
    A, P = eat_cake(T, N, B, u=u)
    n = N
    path = [P[n,0]]
    for t in range(1, T+1):
        i = int(round(path[-1]*N))
        n -= i
        path.append(P[n,t])
    graph_policy(path, b, u)
    return A[-1, 0], path

pol1 = np.array([1, 0, 0, 0, 0])
pol2 = np.array([0, 0, 0, 0, 1])
pol3 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
pol4 = np.array([.4, .3, .2, .1, 0])
b = 0.9
u = lambda x: np.sqrt(x)
print(graph_policy(pol4, b, u))
print(consumption(4))
print(eat_cake(3, 4, 0.9))
print(find_policy(4, 5, 0.9))