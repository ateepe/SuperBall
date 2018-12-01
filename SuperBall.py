import numpy as np
import DisjointSet

default_goals = \
    [False, False, False, False, False, False, False, False, False, False,\
    False, False, False, False, False, False, False, False, False, False,\
    True, True, False, False, False, False, False, False, True, True,\
    True, True, False, False, False, False, False, False, True, True,\
    True, True, False, False, False, False, False, False, True, True,\
    True, True, False, False, False, False, False, False, True, True,\
    False, False, False, False, False, False, False, False, False, False,\
    False, False, False, False, False, False, False, False, False, False] 

default_colors = ['p', 'b', 'y', 'r', 'g']

class SuperBall:

    def __init__(self, nRows=8, nCols=10, goals=default_goals, colors=default_colors, minSetSize=5):
        self.numRows = nRows
        self.numCols = nCols
        self.numTiles = nRows * nCols
        self.minSetSize = minSetSize

        self.numOpenTiles = self.numTiles
        self.gameOver = False

        self.board = np.array(['.' for i in range(self.numTiles)])
        self.goals = np.array([False for i in range(self.numTiles)])
        self.colors = colors

        for pos in goals:
            row, col = pos
            self.goals[row*self.numCols + col] = True

        self.totalScore = 0

        self.spawnTiles(5) # initialize game with 5 tiles

    def StartGame(self):
        self.board = np.array(['.' for i in range(self.numTiles)])
        self.numOpenTiles = self.numTiles
        self.totalScore = 0

        self.spawnTiles(5)

    def canScore(self, row, col):
        index = row * self.numCols + col

        if (not self.goals[index] or self.board[index] == '.'):
            return False

        djSet = self.getDJSet()
        setSize = djSet.getSetSize(index)

        if setSize < self.minSetSize:
            return False

        return True

    def score(self, row, col):
        index = row * self.numCols + col

        if (not self.goals[index] or self.board[index] == '.'):
            return 0

        score = 0
        djSet = self.getDJSet()
        setSize = djSet.getSetSize(index)
        colorValue = self.colors.index(self.board[index]) + 2

        if setSize < self.minSetSize:
            return 0
        
        score = setSize * colorValue

        self.totalScore += score

        scoringSetID = djSet.getSetID(index)
        for i in range(self.numTiles):
            if djSet.getSetID(i) == scoringSetID:
                self.board[i] = '.'
        
        self.numOpenTiles += setSize

        self.spawnTiles(3)

        return score

    def canSwap(self, row1, col1, row2, col2):
        index1 = row1 * self.numCols + col1
        index2 = row2 * self.numCols + col2

        if self.board[index1] == '.' or self.board[index2] == '.':
            return False
        
        return True

    def swap(self, row1, col1, row2, col2):
        index1 = row1 * self.numCols + col1
        index2 = row2 * self.numCols + col2

        if self.board[index1] == '.' or self.board[index2] == '.':
            return False

        temp = self.board[index1]
        self.board[index1] = self.board[index2]
        self.board[index2] = temp

        return self.spawnTiles(5)

    def spawnTiles(self, numTiles):
        if self.numOpenTiles < numTiles:
            self.numOpenTiles -= numTiles
            self.gameOver = True
            return False
        
        for i in range(numTiles):
            index = np.random.randint(0, self.numTiles)
            while self.board[index] != '.':
                index = np.random.randint(0, self.numTiles)

            self.board[index] = self.colors[np.random.randint(0, len(self.colors))]
    
        self.numOpenTiles -= numTiles

        return True

    def getDJSet(self):
        djSet = DisjointSet.DisjointSet(self.numTiles)

        # union vertically
        for r in range(1, self.numRows):
            for c in range(0, self.numCols):
                index = r * self.numCols + c
                upIndex = (r-1) * self.numCols + c
                if (self.board[index] == self.board[upIndex]):
                    djSet.union(index, upIndex)

        # union horizontally
        for c in range(1, self.numCols):
           for r in range(0, self.numRows):
                index = r * self.numCols + c
                leftIndex = index - 1
                if (self.board[index] == self.board[leftIndex]):
                    djSet.union(index, leftIndex)

        return djSet

    def getTile(self, index):
        return self.board[index]

    def isGoal(self, index):
        return self.goals[index]
    
    def print(self):
        for r in range(self.numRows):
            for c in range(self.numCols):
                index = r * self.numCols + c
                print(self.board[index], end=" ")
            print("") # newline
        print("")