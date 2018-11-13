import SuperBall


class RLPlayer:

    def __init__(self, nRows, nCols, goals, colors, minSetSize):
        self.game = SuperBall.SuperBall(nRows, nCols, goals, colors, minSetSize);

        # actions = pick any two tiles and swap them, or any of the goal tiles and try to score
        #           if any action can't be taken, just allow the machine to take it but nothing happens and there is no reward
        # states = the state of the board
        # 
        # since actions of swap and score are different dimenstiosn (two tiles for swap vs one tile for score), should policy and action
        # value data structures be dictionaries to accommodate?
        # 
        # Since so many possible states, should they be added dynamically as encountered rather than initialize all at once? 

    def sarsa(self, num_episdoes):
        pass

    def choose_action(self, state):
        pass

    def update_policy(self, state):
        pass
    
if __name__ == "__main__":
    pass