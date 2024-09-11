# pagerank.py
"""Volume 1: The Page Rank Algorithm.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy.sparse import dok_matrix
from scipy import linalg as la

# Problem 1
def to_matrix(filename, n):
    """Return the nxn adjacency matrix described by datafile.

    Parameters:
        datafile (str): The name of a .txt file describing a directed graph.
        Lines describing edges should have the form '<from node>\t<to node>\n'.
        The file may also include comments.
    n (int): The number of nodes in the graph described by datafile

    Returns:
        A SciPy sparse dok_matrix.
    """
    M = dok_matrix((n,n)) # Create an empty matrix
    with open(filename, 'r') as myfile:
        for line in myfile:
            try:
                i = line.strip().split() # Process the line
                M[int(i[0]), int(i[1])] = 1 # Set the value of the adjacency matrix
            except ValueError as e:
                continue
    return M
            


# Problem 2
def calculateK(A,N):
    """Compute the matrix K as described in the lab.

    Parameters:
        A (ndarray): adjacency matrix of an array
        N (int): the datasize of the array

    Returns:
        K (ndarray)
    """
    # Compute the modified adjacency matrix for the sinks
    for i,r in enumerate(A):
        if np.allclose(r, np.zeros(N)):
            A[i] = np.ones(N)
    #Compute the diagonal matrix D
    D = np.array([sum(r) for r in A])
    # Compute K using array broadcasting
    return (A.T/D)

# Problem 3
def iter_solve(adj, N=None, d=.85, tol=1E-5):
    """Return the page ranks of the network described by 'adj'.
    Iterate through the PageRank algorithm until the error is less than 'tol'.

    Parameters:
        adj (ndarray): The adjacency matrix of a directed graph.
        N (int): Restrict the computation to the first 'N' nodes of the graph.
            If N is None (default), use the entire matrix.
        d (float): The damping factor, a float between 0 and 1.
        tol (float): Stop iterating when the change in approximations to the
            solution is less than 'tol'.

    Returns:
        The approximation to the steady state.
    """
    # Handle the case when N=None
    if N == None:
        N =len(adj)
    # Work with the upper N X N matrix
    A = adj[:N, :N]
    K = calculateK(A, N)
    # Initialize p0 as a random vector
    p0 = np.random.random(N)
    # Calculate p1 using equation 14.3
    p1 = d*K@p0 + (1-d)/N * np.ones(N)
    # Continue with the iteration until within tolerance
    while la.norm(p1-p0) > tol:
        p0 = p1
        p1 = d*K@p0 + (1-d)/N * np.ones(N)
        
    return p1


# Problem 4
def eig_solve(adj, N=None, d=.85):
    """Return the page ranks of the network described by 'adj'. Use SciPy's
    eigenvalue solver to calculate the steady state of the PageRank algorithm

    Parameters:
        adj (ndarray): The adjacency matrix of a directed graph.
        N (int): Restrict the computation to the first 'N' nodes of the graph.
            If N is None (default), use the entire matrix.
        d (float): The damping factor, a float between 0 and 1.
        tol (float): Stop iterating when the change in approximations to the
            solution is less than 'tol'.

    Returns:
        The approximation to the steady state.
    """
    # Same initial process as prob 3
    if N == None:
        N =len(adj)
    A = adj[:N, :N]
    K = calculateK(A, N)
    # Create E a matrix of ones
    E = np.ones((N,N))
    # Compute B using the formula given
    B = d*K + (1-d)/N * E
    # Compute the eigen values/vectors and get the vector that corresponds with 1
    val, vec = la.eig(B)
    i = np.argmax(val)
    p = vec[:,i]
    # Normalize
    return p / np.sum(p)

# Problem 5
def team_rank(filename='ncaa2013.csv'):
    """Use iter_solve() to predict the rankings of the teams in the given
    dataset of games. The dataset should have two columns, representing
    winning and losing teams. Each row represents a game, with the winner on
    the left, loser on the right. Parse this data to create the adjacency
    matrix, and feed this into the solver to predict the team ranks.

    Parameters:
        filename (str): The name of the data file.
    Returns:
        ranks (list): The ranks of the teams from best to worst.
        teams (list): The names of the teams, also from best to worst.
    """
    teams = set()
    games = list()
    # Read the input, add the save the necessary information
    with open(filename, 'r') as myfile:
        myfile.readline()
        for line in myfile:
            game = line.strip().split(',')
            teams.add(game[0])
            teams.add(game[1])
            games.append(game)
    teams = np.array(list(teams))
    N = len(teams)
    # Create a dictionary of the teams corresponding to a numeric value
    team_dict = {name:i for i, name in enumerate(teams)}
    # Create the adjacency matrix
    A = dok_matrix((N,N))
    for g in games:
        A[team_dict[g[1]], team_dict[g[0]]] = 1
    # Calulate the rank
    p = iter_solve(A.toarray(), d=0.7)
    # Return the ranks and teams in order
    return p[np.argsort(p)][::-1], teams[np.argsort(p)][::-1]

A = to_matrix('matrix.txt', 8)
p = eig_solve(A.toarray())
print(p)