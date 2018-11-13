import SuperBall


class RLPlayer:

    def __init__(self, nRows, nCols, goals, colors, minSetSize):
        self.game = SuperBall.SuperBall(nRows, nCols, goals, colors, minSetSize);

    def sarsa(self, num_episdoes):
        pass

    def choose_action(self, state):
        pass

    def update_policy(self, state):
        pass
    
if __name__ == "__main__":
    pass