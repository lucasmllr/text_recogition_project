
class DisjointSet():
    '''
    A class for a disjoint set data structure to manage the blob labels.
    Labels must be added in rising order starting from 1 and no integer value can be skipped.

    inspired by: https://gist.github.com/bruceoutdoors/3b99fd1c266b3718e1084352dbcb19c6
    Contrary to the above sets are not joint according to their rank, but according to whose root label is smaller,
    so that set representatives are always the smallest labels.
    '''

    def __init__(self, n_labels=1):
        """
        initializing a single set with label 0
        """
        self.n = n_labels
        self.parents = [i for i in range(n_labels)]


    def exists(self, l):
        """
        Checks whether label l exists.

        Args:
            l (int): label to be checked

        Returns:
            bool indicating existence
        """
        n = len(self.parents)
        if l >= n:
            return False
        else:
            return True


    def add(self, l):
        """
        Adds a label l if it does not yet exist.

        Args:
            l (int): label to be added
        """

        if self.exists(l):
            raise ValueError('label {} already exists'.format(l))

        self.n += 1
        self.parents.append(l)


    def find_root(self, l):
        """
        Finds the root of the set that label l belongs to.

        Args:
            l (int): label whose root to find

        Returns:
            the root label
        """

        if not self.exists(l):
            raise ValueError('label {} does not exist'.format(l))

        # if leaf is not connected directly to the root or is the root itself, connect it directly to the root
        # (path compression)
        if self.parents[l] != l:
            self.parents[l] = self.find_root(self.parents[l])

        return self.parents[l]


    def connected(self, l, m):
        """
        Checks whether labels l and m belong to the same set.

        Args:
            l (int): one label
            m (int): another label

        Returns:
            bool indicating whether it is the case
        """

        return self.find_root(l) == self.find_root(m)


    def unite(self, l, m):
        """
        Joins the sets of labels l and m keeping as a root the smallest element in both sets.

        Args:
            l (int): label in set one
            m (int): label in set two
        """

        if self.connected(l, m):
            return

        root_l = self.find_root(l)
        root_m = self.find_root(m)

        if root_l < root_m:
            print('\nroot of {}: {} becomes root of {}'.format(l, root_l, m))
            print(self.get_equivalents())
            self.parents[root_m] = root_l
        else:
            print('\nroot of {}: {} becomes root of {}'.format(m, root_m, l))
            print(self.get_equivalents())
            self.parents[root_l] = root_m


    def next(self):

        return self.n


    def get_equivalents(self):

        eq = {}
        for l in range(len(self.parents)):
            #if l != self.parents[l]:
            eq[l] = self.find_root(l)

        return eq


    def final_labels(self):
        """
        Returns a list of remaining labels after eliminating equivalents.
        In the returned list elements are labels now, not indices as before!
        """
        return [self.parents[i] for i in range(1, self.n) if i == self.parents[i]]


if __name__ == "__main__":

    labels = DisjointSet()
    labels.add(1)
    labels.add(2)
    labels.add(3)
    labels.add(4)
    labels.add(5)
    labels.add(6)

    print(labels.connected(1, 2))
    labels.unite(1, 2)
    print(labels.connected(1, 2))
    print(labels.find_root(2))

    labels.unite(2, 5)
    print(labels.find_root(5))



