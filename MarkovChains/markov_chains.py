# markov_chains.py
"""Volume II: Markov Chains.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la


# Problem 1
def random_chain(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.
    """
    A = np.random.random((n,n))
    y = [sum(A[:, i]) for i in range(n)]
    return A/y

# Problem 2
def forecast(days):
    """Forecast tomorrow's weather given that today is hot."""
    transition = np.array([[0.7, 0.6], [0.3, 0.4]])

    # Sample from a binomial distribution to choose a new state.
    l = [0]
    for i in range(days):
        l.append(np.random.binomial(1, transition[1,l[i]]))
    return l[1:]


# Problem 3
def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    transition = np.array([[0.5, 0.3, 0.1, 0],
                           [0.3, 0.3, 0.3, 0.3],
                           [0.2, 0.3, 0.4, 0.5],
                           [0, 0.1, 0.2, 0.2]])
    l = [1]
    for i in range(days):
        n = np.random.multinomial(1, transition[:,l[i]])
        l.append(np.where(n==1)[0][0])
    return l[1:]

# Problem 4
def steady_state(A, tol=1e-12, N=40):
    """Compute the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """
    x = np.random.random(len(A))
    x = x/sum(x)
    for k in range(N):
        y = x.copy()
        x = A@x
        if la.norm(x-y) < tol:
            break
    if la.norm(x-y) > tol:
        raise ValueError("A**k does not converge")
    return x

# Problems 5 and 6
class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        (what attributes do you need to keep track of?)

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print(yoda.babble())
        The dark side of loss is a path as one with you.
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        sentences = []
        with open(filename, 'r') as file:
            sentences = file.read().split('\n')
        if len(sentences[-1]) == 0:
            sentences = sentences[:-1]
        self.words = []
        for i in range(len(sentences)):
            self.words += sentences[i].split(' ')
        self.words = list(set(self.words))
        count = len(self.words)
        self.transition = np.zeros((count+2, count+2))
        for i in range(len(sentences)):
            s = sentences[i].split(' ')
            self.transition[self.words.index(s[0])+1, 0] += 1
            for j in range(len(s)-1):
                self.transition[self.words.index(s[j+1])+1, self.words.index(s[j])+1] += 1
            self.transition[-1, self.words.index(s[-1])+1] += 1
        self.transition[-1,-1] += 1.
        y = np.array([sum(self.transition[:, i]) for i in range(len(self.transition))])
        y[y==0.0] = 1
        self.transition = self.transition/y

    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """
        l = [0]
        i = 0
        while i == 0: 
            n = np.random.multinomial(1, self.transition[:,l[-1]])
            l.append(np.where(n==1)[0][0])
            if n[-1] == 1:
                break
        s = [self.words[i-1] for i in l[1:-1]]
        return ' '.join(s)


