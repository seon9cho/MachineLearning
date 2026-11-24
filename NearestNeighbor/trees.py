# trees.py
"""Volume 2: Trees.
<Name>
<Class>
<Date>
"""

class SinglyLinkedListNode:
    """Simple singly linked list node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A very simple singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a Node containing 'data' to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

def iterative_search(linkedlist, data):
    """Search 'linkedlist' iteratively for a node containing 'data'.
    If there is no such node in the list, or if the list is empty,
    raise a ValueError.

    Inputs:
        linkedlist (SinglyLinkedList): a linked list.
        data: the data to search for in the list.

    Returns:
        The node in 'linkedlist' containing 'data'.
    """
    current = linkedlist.head
    while current is not None:
        if current.value == data:
            return current
        current = current.next
    raise ValueError(str(data) + " is not in the list.")

# Problem 1
def recursive_search(linkedlist, data):
    """Search 'linkedlist' recursively for a node containing 'data'.
    If there is no such node in the list, or if the list is empty,
    raise a ValueError.

    Inputs:
        linkedlist (SinglyLinkedList): a linked list object.
        data: the data to search for in the list.

    Returns:
        The node in 'linkedlist' containing 'data'.
    """
    temp = linkedlist.head
    end = linkedlist.tail
    if linkedlist.head == None:
        raise ValueError(str(data) + " is not in the list.")
    def search_recursion(node, data, end_node):
        if node.value == data:
            return node
        elif node == end_node:
            raise ValueError(str(data) + " is not in the list.")
        else:
            return search_recursion(node.next, data, end_node)
    return search_recursion(temp, data, end)

class BSTNode:
    """A Node class for Binary Search Trees. Contains some data, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the data attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary Search Tree data structure class.
    The 'root' attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing 'data'. If there is no such node
        in the tree, or if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            'data' is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Step to the left.
                return _step(current.left)
            else:                                   # Step to the right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing 'data' at the appropriate location.
        Do not allow for duplicates in the tree: if there is already a node
        containing 'data' in the tree, raise a ValueError.

        Example:
            >>> b = BST()       |   >>> b.insert(1)     |       (4)
            >>> b.insert(4)     |   >>> print(b)        |       / \
            >>> b.insert(3)     |   [4]                 |     (3) (6)
            >>> b.insert(6)     |   [3, 6]              |     /   / \
            >>> b.insert(5)     |   [1, 5, 7]           |   (1) (5) (7)
            >>> b.insert(7)     |   [8]                 |             \
            >>> b.insert(8)     |                       |             (8)
        """
        def _step(node, data):
            if node.value == data:
                raise ValueError(str(data) + " already exists.")
            elif node.value > data:
                if node.left == None:
                    new_node = BSTNode(data)
                    node.left = new_node
                    node.left.prev = node
                else:
                    _step(node.left, data)
            else:
                if node.right == None:
                    new_node = BSTNode(data)
                    node.right = new_node
                    node.right.prev = node
                else:
                    _step(node.right, data)

        if self.root == None:
            new_node = BSTNode(data)
            self.root = new_node
        else:
            _step(self.root, data)

    # Problem 3
    def remove(self, data):
        """Remove the node containing 'data'. Consider several cases:
            1. The tree is empty
            2. The target is the root:
                a. The root is a leaf node, hence the only node in the tree
                b. The root has one child
                c. The root has two children
            3. The target is not the root:
                a. The target is a leaf node
                b. The target has one child
                c. The target has two children
            If the tree is empty, or if there is no node containing 'data',
            raise a ValueError.

        Examples:
            >>> print(b)        |   >>> b.remove(1)     |   [5]
            [4]                 |   >>> b.remove(7)     |   [3, 8]
            [3, 6]              |   >>> b.remove(6)     |
            [1, 5, 7]           |   >>> b.remove(4)     |
            [8]                 |   >>> print(b)        |
        """
        target = self.find(data)

        def left_most(node):
            if node.left == None:
                return node
            else:
                return left_most(node.left)

        if target == self.root:
            if self.root.left == None and self.root.right == None:
                self.root = None
            elif self.root.left != None and self.root.right != None:
                predecessor = left_most(self.root.right)
                if predecessor.prev.right == predecessor:
                    predecessor.left = self.root.left
                    self.root.left.prev = predecessor
                    self.root = predecessor
                else:
                    predecessor.left = self.root.left
                    predecessor.right = self.root.right
                    predecessor.prev.left = None
                    self.root.left.prev = predecessor
                    self.root.right.prev = predecessor
                    self.root = predecessor
            else:
                if self.root.right == None:
                    self.root = target.left
                    self.root.prev = None
                else:
                    self.root = target.right
                    self.root.prev = None

        else:
            if target.left == None and target.right == None:
                if target.prev.value > target.value:
                    target.prev.left = None
                else:
                    target.prev.right = None
            elif target.left != None and target.right != None:
                predecessor = left_most(target.right)
                if predecessor.prev.right == predecessor:
                    predecessor.left = target.left
                    target.left.prev = predecessor
                else:
                    predecessor.left = target.left
                    predecessor.right = target.right
                    predecessor.prev.left = None
                    target.left.prev = predecessor
                    target.right.prev = predecessor
                if target.prev.value > target.value:
                    target.prev.left = predecessor
                    predecessor.prev = target.prev
                else:
                    target.prev.right = predecessor
                    predecessor.prev = target.prev
            else:
                if target.right == None:
                    if target.prev.value > target.value:
                        target.prev.left = target.left
                        target.left.prev = target.prev
                    else:
                        target.prev.right = target.left
                        target.left.prev = target.prev
                else:
                    if target.prev.value > target.value:
                        target.prev.left = target.right
                        target.right.prev = target.prev
                    else:
                        target.prev.right = target.right
                        target.right.prev = target.prev                    


    def __str__(self):
        """String representation: a hierarchical view of the BST.
        Do not modify this method, but use it often to test this class.
        (this method uses a depth-first search; can you explain how?)

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed out
                (2) (5)    [2, 5]       by depth levels. The edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """

        if self.root is None:
            return "[]"
        str_tree = [list() for i in range(_height(self.root) + 1)]
        visited = set()

        def _visit(current, depth):
            """Add the data contained in 'current' to its proper depth level
            list and mark as visited. Continue recusively until all nodes have
            been visited.
            """
            str_tree[depth].append(current.value)
            visited.add(current)
            if current.left and current.left not in visited:
                _visit(current.left, depth+1)
            if current.right and current.right not in visited:
                _visit(current.right, depth+1)

        _visit(self.root, 0)
        out = ""
        for level in str_tree:
            if level != list():
                out += str(level) + "\n"
            else:
                break
        return out


class AVL(BST):
    """AVL Binary Search Tree data structure class. Inherits from the BST
    class. Includes methods for rebalancing upon insertion. If your
    BST.insert() method works correctly, this class will work correctly.
    Do not modify.
    """
    def _checkBalance(self, n):
        return abs(_height(n.left) - _height(n.right)) >= 2

    def _rotateLeftLeft(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n == self.root:
            self.root = temp
        return temp

    def _rotateRightRight(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n == self.root:
            self.root = temp
        return temp

    def _rotateLeftRight(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotateLeftLeft(n)

    def _rotateRightLeft(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotateRightRight(n)

    def _rebalance(self,n):
        """Rebalance the subtree starting at the node 'n'."""
        if self._checkBalance(n):
            if _height(n.left) > _height(n.right):
                if _height(n.left.left) > _height(n.left.right):
                    n = self._rotateLeftLeft(n)
                else:
                    n = self._rotateLeftRight(n)
            else:
                if _height(n.right.right) > _height(n.right.left):
                    n = self._rotateRightRight(n)
                else:
                    n = self._rotateRightLeft(n)
        return n

    def insert(self, data):
        """Insert a node containing 'data' into the tree, then rebalance."""
        BST.insert(self, data)
        n = self.find(data)
        while n:
            n = self._rebalance(n)
            n = n.prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() has been disabled for this class.")

def _height(current):
    """Calculate the height of a given node by descending recursively until
    there are no further child nodes. Return the number of children in the
    longest chain down.

    This is a helper function for the AVL class and BST.__str__().
    Abandon hope all ye who modify this function.

                                node | height
    Example:  (c)                  a | 0
              / \                  b | 1
            (b) (f)                c | 3
            /   / \                d | 1
          (a) (d) (g)              e | 0
                \                  f | 2
                (e)                g | 0
    """
    if current is None:
        return -1
    return 1 + max(_height(current.right), _height(current.left))


# Problem 4
def prob4():
    """Compare the build and search speeds of the SinglyLinkedList, BST, and
    AVL classes. For search times, use iterative_search(), BST.find(), and
    AVL.find() to search for 5 random elements in each structure. Plot the
    number of elements in the structure versus the build and search times.
    Use log scales if appropriate.
    """
    import numpy as np
    import time
    import random
    from matplotlib import pyplot as plt

    content = []
    with open("english.txt", 'r') as file:
        for line in file:
            content.append(line.strip('\n'))

    buildSLL = []
    buildBST = []
    buildAVL = []
    searchSLL = []
    searchBST = []
    searchAVL = []
    x = [2**i for i in range(3,12)]

    for n in x:
        s = SinglyLinkedList()
        b = BST()
        a = AVL()
        randindex = random.sample(range(0,len(content)), n)
        randset = [content[i] for i in randindex]
        t1 = time.time()
        for i in range(n):
            s.append(randset[i])
        t2 = time.time()
        for i in range(n):
            b.insert(randset[i])
        t3 = time.time()
        for i in range(n):
            a.insert(randset[i])
        t4 = time.time()
        buildSLL.append(t2-t1)
        buildBST.append(t3-t2)
        buildAVL.append(t4-t3)

        findindex = random.sample(range(0, len(randset)), 5)
        t5 = time.time()
        for i in findindex:
            iterative_search(s, randset[i])
        t6 = time.time()
        for i in findindex:
            b.find(randset[i])
        t7 = time.time()
        for i in findindex:
            a.find(randset[i])
        t8 = time.time()
        searchSLL.append(t6-t5)
        searchBST.append(t7-t6)
        searchAVL.append(t8-t7)

    plt.ion()
    ax1 = plt.subplot(121)
    ax1.loglog(x, buildSLL, 'b.-', label='SinglyLinkedList', basex=2, basey=2)
    ax1.loglog(x, buildBST, 'r.-', label='BST', basex=2, basey=2)
    ax1.loglog(x, buildAVL, 'g.-', label='AVL', basex=2, basey=2)
    ax1.set_title("Build time")
    ax1.set_xlabel('n')
    ax1.set_ylabel('seconds')
    ax1.legend(loc="upper left")

    ax2 = plt.subplot(122)
    ax2.loglog(x, searchSLL, 'b.-', basex=2, basey=2)
    ax2.loglog(x, searchBST, 'r.-', basex=2, basey=2)
    ax2.loglog(x, searchAVL, 'g.-', basex=2, basey=2)
    ax2.set_title("Search time")
    ax2.set_xlabel('n')




