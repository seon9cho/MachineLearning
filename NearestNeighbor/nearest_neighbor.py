# nearest_neighbor.py
"""Volume 2: The Nearest Neighbor Problem.
<Name>
<Class>
<Date>
"""
import numpy as np
import scipy as sp
from numpy import linalg as la
import sys
sys.path.insert(1, "../Trees")
from trees import BSTNode
from trees import BST
from scipy.spatial import KDTree

# Problem 1
def metric(x, y):
    """Return the Euclidean distance between the 1-D arrays 'x' and 'y'.

    Raises:
        ValueError: if 'x' and 'y' have different lengths.

    Example:
        >>> metric([1,2],[2,2])
        1.0
        >>> metric([1,2,1],[2,2])
        ValueError: Incompatible dimensions.
    """
    if len(x) != len(y):
        raise ValueError("Incompatible dimensions")
    x = np.array(x)
    y = np.array(y)
    return la.norm(y - x)

# Problem 2
def exhaustive_search(data_set, target):
    """Solve the nearest neighbor search problem exhaustively. Check the distances between 'target' and each point in 'data_set'. Use the Euclidean metric to calculate distances.

    Parameters:
        data_set ((m,k) ndarray): An array of m k-dimensional points.
        target ((k,) ndarray): A k-dimensional point to compare to 'dataset'.

    Returns:
        ((k,) ndarray) the member of 'data_set' that is nearest to 'target'.
        (float) The distance from the nearest neighbor to 'target'.
    """
    nearest_dist = metric(data_set[0], target)
    nearest_pt = data_set[0]
    for X in data_set:
        dist = metric(X, target)
        if nearest_dist > dist:
            nearest_dist = dist
            nearest_pt = X
    return nearest_pt, nearest_dist


# Problem 3: Write a KDTNode class.
class KDTNode(BSTNode):
    def __init__(self, data):
        if type(data) != np.ndarray:
            raise TypeError("Input must be numpy.ndarray")
        BSTNode.__init__(self, data)
        self.axis = None

# Problem 4: Finish implementing this class by overriding
#            the __init__(), insert(), and remove() methods.
class KDT(BST):
    """A k-dimensional binary search tree object. Used to solve the nearest neighbor problem efficiently.

    Attributes:
        root (KDTNode): The root node of the tree. Like all other nodes in the
            tree, the root houses data as a NumPy array.
        k (int): The dimension of the tree (the 'k' of the k-d tree).
    """

    def find(self, data):
        """Return the node containing 'data'. If there is no such node in the tree, or if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing 'data' is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.axis] < current.value[current.axis]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursion on the root of the tree.
        return _step(self.root)

    def insert(self, data):
        """Insert a new node containing 'data' at the appropriate location.
        Return the new node. This method should be similar to BST.insert().
        """
        def _step(node, data):
            if np.allclose(node.value, data):
                raise ValueError(str(data) + " already exists.")
            elif data[node.axis] < node.value[node.axis]:
                if node.left == None:
                    new_node = KDTNode(data)
                    if node.axis == len(node.value) - 1:
                        new_node.axis = 0
                    else:
                        new_node.axis = node.axis + 1
                    node.left = new_node
                    node.left.prev = node
                else:
                    _step(node.left, data)
            else:
                if node.right == None:
                    new_node = KDTNode(data)
                    if node.axis == len(node.value) - 1:
                        new_node.axis = 0
                    else:
                        new_node.axis = node.axis + 1
                    node.right = new_node
                    node.right.prev = node
                else:
                    _step(node.right, data)

        if self.root == None:
            new_node = KDTNode(data)
            new_node.axis = 0
            self.root = new_node
        else:
            _step(self.root, data)

    def remove(*args, **kwargs):
        raise NotImplementedError("remove() has been disabled for this class.")


# Problem 5
def nearest_neighbor(data_set, target):
    """Use your KDT class to solve the nearest neighbor problem.

    Parameters:
        data_set ((m,k) ndarray): An array of m k-dimensional points.
        target ((k,) ndarray): A k-dimensional point to compare to 'dataset'.

    Returns:
        The point in the tree that is nearest to 'target' ((k,) ndarray).
        The distance from the nearest neighbor to 'target' (float).
    """
    k = KDT()
    for n in data_set:
        k.insert(n)

    def KDTsearch(current, neighbor, distance):
        """The actual nearest neighbor search algorithm.

        Parameters:
            current (KDTNode): the node to examine.
            neighbor (KDTNode): the current nearest neighbor.
            distance (float): the current minimum distance.

        Returns:
            (ndarray): The new nearest neighbor in the tree.
            (float): The new minimum distance.
        """
        if current == None:
            return neighbor, distance
        if metric(current.value, target) < distance:
            neighbor = current
            distance = metric(current.value, target)
        if target[current.axis] < current.value[current.axis]:
            neighbor, distance = KDTsearch(current.left, neighbor, distance)
            if target[current.axis] + distance >= current.value[current.axis]:
                neighbor, distance = KDTsearch(current.right, neighbor, distance)
        else:
            neighbor, distance = KDTsearch(current.right, neighbor, distance)
            if target[current.axis] - distance <= current.value[current.axis]:
                neighbor, distance = KDTsearch(current.left, neighbor, distance)
        return neighbor, distance

    n,d = KDTsearch(k.root, k.root, metric(k.root.value, target))
    return n.value, d

# Problem 6
class KNeighborsClassifier(object):
    """A k-nearest neighbors classifier object. Uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """

    def __init__(self, data, labels):
        """Initialize the training set and labels. Construct the KDTree from
        the training data.

        Parameters:
            data (ndarray): Training data.
            labels (ndarray): Corresponding labels for the training data.
        """
        self.tree = KDTree(data)
        self.data = data
        self.labels = labels

    def predict(self, testpoints, k):
        """Predict the label of a new data point by finding the k-nearest
        neighbors.

        Parameters:
            testpoints (ndarray): New data point(s) to label.
            k (int): Number of neighbors to find.
        """
        def classify(neighbors):
            l = []
            for i in neighbors:
                l.append(self.labels[i])
            return sp.stats.mode(l)[0][0]

        labels = []
        for p in testpoints:
            distances, neighbors = self.tree.query(p, k=k)
            labels.append(classify(neighbors))
        return labels



