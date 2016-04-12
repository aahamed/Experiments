"""
Author: Aadil Ahamed
Created: 4/1/2016
kd_tree.py: Implementation of a kd-tree
"""

class point:
    """ Class to represent a k-dimensional point """

    def __init__(self, coords):
        self.coords = coords


class kd_node:
    """ Class to represent a node in the KD Tree"""

    def __init__(self, axis, value, left=None, right=None, points=None):
        """
        axis: splitting axis
        value: splitting value
        left: left subtree
        right: right subtree
        points: list of points (only present in leaf nodes)
        """
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.points = points


class kd_tree:
    " Class to represent a kd-tree"
    
    MAX_DEPTH = 0
    AXIS_MAP = {0: 'X', 1: 'Y', 2: 'Z'}
    
    def __init__(self, data, max_depth=float("inf")):
        self.k = len(data[0])
        self.MAX_DEPTH = max_depth
        self.root = self.build_tree(data, 0)

    @staticmethod
    def sdistance(p, q):
        """TODO"""
        dist = 0
        for i in range(len(p)):
            dist += (p[i] - q[i])**2
        return dist

    def build_tree(self, data, depth):
        """
        Build a balanced KD-Tree using data
        data: data stored in kd-tree
        depth: current depth of the tree
        """
        axis = depth % self.k   # calculate splitting axis based on depth
        # Case 1: data list is empty
        if len(data) == 0:
            return None
        # Case 2: Tree has reached MAX_DEPTH
        elif depth == self.MAX_DEPTH:
            return kd_node(axis, None, None, None, data)
        # Case 3: data list has 1 element
        elif len(data) == 1:
            return kd_node(axis, None, None, None, [data[0]])
        else:
            sorted_data = sorted(data, key=lambda point: point[axis])
            index_median = len(sorted_data)//2  # Find median
            left = sorted_data[0:index_median]
            right = sorted_data[index_median:]
            left_child = self.build_tree(left, depth+1)   # recursively create left and right subtree by splitting on the median
            right_child = self.build_tree(right, depth+1)
            return kd_node(axis, sorted_data[index_median][axis], left_child, right_child)

    @staticmethod
    def nns_batch(data, queries, max_depth=float("inf")):
        """ Nearest Neighbor Search on a batch of queries """
        tree = kd_tree(data, max_depth)
        res = []
        for query in queries:
            res.append(tree.nns(query))
        return res

    def nns(self, query):
        """Top Level Nearest Neighbor Search"""
        return self.nns_rec(query, self.root, None, float("inf"))

    @staticmethod
    def min_index(array):
        min_ind = 0
        for i in range(len(array)):
            if array[i] < array[min_ind]:
                min_ind = i
        return min_ind

    def nns_rec(self, query, node, best_point, best_dist):
        """ 
        Nearest Neighbor Search
        query: query point
        node: initial node
        best_point: current closest neighbor
        best_dist: current best_distance
        returns query point's nearest neighbor
        """
        # Case 1: Node is None
        if node is None:
            return None

        # Case 2: Node is a Leaf Node
        if node.left is None and node.right is None:
            # There may be more than 1 point in leaf node
            min_ind = 0
            min_dist = self.sdistance(query, node.points[0])
            for i in range(1, len(node.points)):
                dist = self.sdistance(query, node.points[i]) 
                if dist < min_dist:
                    min_dist = dist
                    min_ind = i
            if min_dist < best_dist:
                return node.points[min_ind]     # return point that is closest to query point
        else:
            if query[node.axis] < node.value:
                best_point = self.nns_rec(query, node.left, best_point, best_dist)
                best_dist = self.sdistance(query, best_point)
                if query[node.axis] + best_dist >= node.value:
                    best_point = self.nns_rec(query, node.right, best_point, best_dist)
            else:
                best_point = self.nns_rec(query, node.right, best_point, best_dist)
                best_dist = self.sdistance(query, best_point)
                if query[node.axis] - best_dist <= node.value:
                    best_point = self.nns_rec(query, node.left, best_point, best_dist)

        return best_point

    def print_in_order(self, node):
        if node is None:
            return
        else:
            self.print_in_order(node.left)
            if node.points is not None:
                print("leaf:", node.points)
            else:
                print("{} = {}".format(self.AXIS_MAP[node.axis], node.value))
            self.print_in_order(node.right)


def test_kdtree():
    print("Testcase 1:")
    a = [ [1, 1], [2, 2], [1, 3], [5, 1], [6, 8], [3, 2] ]
    tree = kd_tree(a)
    tree.print_in_order(tree.root)
    print("\nTestcase 2:")
    b = [ [1, 1], [2, 2], [3, 3], [4, 4], [5,5] ]
    tree = kd_tree(b)
    tree.print_in_order(tree.root)
    print("\nTestcase 3:")
    c = [ [1, 1], [2, 2], [1, 3], [5, 1], [6, 8], [3, 2] ]
    tree = kd_tree(c, 2)
    tree.print_in_order(tree.root)


def main():
    test_kdtree()


if __name__ == "__main__":
    main()
