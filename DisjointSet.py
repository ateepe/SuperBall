import numpy as np


class DisjointSet:

    def __init__(self, n):
        self.links = np.array([-1 for i in range(n)]);
        self.sizes = np.array([0 for i in range(n)]);

    def getSetID(self, element):
        root = element;
        while self.links[root] != -1:
            root = self.links[root];

        return root;

    def union(self, e1, e2):
        setID1 = self.getSetID(e1);
        setID2 = self.getSetID(e2);

        if setID1 == setID2:
            return setID1;

        # first set is larger and becomes parent
        if (self.sizes[setID1] > self.sizes[setID2]):
            self.links[setID2] = setID1;
            self.sizes[setID1] += self.sizes[setID2];
            self.sizes[setID2] = 0;
            return setID1;
        
        # second set is larger (or tied)
        else:
            self.links[setID1] = setID2;
            self.sizes[setID2] += self.sizes[setID1];
            self.sizes[setID1] = 0;
            return setID2;

    def getSetSize(self, element):
        return self.sizes[self.getSetID(element)];