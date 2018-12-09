import SuperBall
import numpy as np
import estimatorModel
import BoardScorer as bs

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import Sequential, load_model

class RLPlayer:

    def __init__(self, nRows=8, nCols=10, goals=SuperBall.default_goals, colors=SuperBall.default_colors, minSetSize=5):
        self.game = SuperBall.SuperBall(nRows, nCols, goals, colors, minSetSize);
        self.discount_factor = 0.9

        # actions = pick any two tiles and swap them, or any of the goal tiles and try to score
        #           if any action can't be taken, just allow the machine to take it but nothing happens and there is no reward
        # states = the state of the board
        # 
        # since actions of swap and score are different dimenstiosn (two tiles for swap vs one tile for score), should policy and action
        # value data structures be dictionaries to accommodate?
        # 
        # Since so many possible states, should they be added dynamically as encountered rather than initialize all at once? 


        # Load the saved model from a file
        try:
            self.nn = load_model("RLPlayer2.h5")
        except:
            self.nn = estimatorModel.define_model()


    def generate_episode(self, self_play=True):
        self.game.StartGame()
        
        saved_afterstate = None
        iter_count = 0

        while (not self.game.gameOver):

            actions, afterstates, action_types = self.get_actions(self.game.board)
            # self.game.print()
            
            # Evaluate each action's value from self-play or otherwise
            if (self_play):
                channels = estimatorModel.split_into_channels(afterstates)
                action_values = self.nn.predict(channels)
            else:
                action_values = np.zeros(len(afterstates))
                for i in range(len(afterstates)):
                    score, score_pos = bs.analyze_board(afterstates[i])
                    action_values[i] = score
            
            # If there are fewer than 5 open tiles, all swaps yield GAME OVER
            if (self.game.numOpenTiles < 5):
                for a in range(len(actions)):
                    if (action_types[a] == 'swap'):
                        action_values[a] = -1

            best_index = action_values.argmax()
            best_action = actions[best_index]
            # print(action_values[best_index])

            # All actions lead to game over state
            if (action_values[best_index] == -1):
                self.game.gameOver = True

            # Swap or score the tiles based on chosen action
            elif (action_types[best_index] == 'swap'):
                r1 = best_action[0][0]
                c1 = best_action[0][1]
                r2 = best_action[1][0]
                c2 = best_action[1][1]
                self.game.swap(r1, c1, r2, c2)
            elif (action_types[best_index] == 'score'):
                r = best_action[0]
                c = best_action[1]
                self.game.score(r, c)

            new_afterstate = np.copy(self.game.board)

            # Give large negative reward for losing
            if (self.game.gameOver):
                reward = -1000
            else:
                reward = 0
            

            # Find the target for the value function
            channels = estimatorModel.split_into_channels([new_afterstate])
            v_new_afterstate = self.nn.predict(channels)
            v_target = reward + self.discount_factor * v_new_afterstate

            # Determine the saved afterstate value
            if (iter_count > 0):
                channels = estimatorModel.split_into_channels([saved_afterstate])
                v_saved_afterstate = self.nn.predict(channels)
                player.nn.fit(channels, [v_target], epochs=1, batch_size=1)
            else:
                v_saved_afterstate = 0

            # print('target:', v_target, 'v_sa:', v_saved_afterstate)
            # if (iter_count > 1):
            #     exit()

            

            saved_afterstate = new_afterstate

            iter_count += 1

        print('Final score:', self.game.totalScore)
        # print('DONE')

        # Sound the bell!
        print('\007')
        return


    def get_actions(self, board):
        actions = []
        afterstates = []
        action_types = []

        # Try all of the possible swaps
        for index1 in range(0, self.game.numTiles):
            for index2 in range(index1+1, self.game.numTiles):
                if (board[index1] != '.' and board[index2] != '.'):
                    r1 = int(index1 // self.game.numCols)
                    c1 = int(index1 % self.game.numCols)
                    r2 = int(index2 // self.game.numCols)
                    c2 = int(index2 % self.game.numCols)
                    swapped_board = np.copy(board)
                    swapped_board[index1] = board[index2]
                    swapped_board[index2] = board[index1]
                    actions.append( ((r1, c1), (r2, c2)) )
                    afterstates.append(swapped_board)
                    action_types.append('swap')

        # Try all of the possible scores
        djSet = self.game.getDJSet()

        for i in range(0, self.game.numTiles):
            if (self.game.isGoal(i) and board[i] != '.'):
                setID = djSet.getSetID(i)
                setSize = djSet.getSetSize(i)
                if (setSize >= self.game.minSetSize):
                    scored_board = np.copy(board)
                    for j in range(0, self.game.numTiles):
                        if (djSet.getSetID(j) == setID):
                            scored_board[j] = '.'
                    r = int(i // self.game.numCols)
                    c = int(i % self.game.numCols)
                    actions.append( (r, c) )
                    afterstates.append(scored_board)
                    action_types.append('score')

        return actions, afterstates, action_types
        exit()

    

    # TODO: train network on observed data
    def train_network(self):
        pass

if __name__ == "__main__":
    player = RLPlayer()
    
    while True:
        player.generate_episode()
        player.nn.save("RLPlayer2.h5")
        # exit()
    
    exit()

    # Generate episodes until the program is killed
    while True:

        player.generate_episode()
        
        # Generate a few episodes at a time so it can train on a larger set of data
        for i in range(0, 1):
            boards, labels = player.generate_episode()
            train_boards = train_boards + boards
            train_labels = train_labels + labels

        train_features = estimatorModel.split_into_channels(train_boards)
        
        player.nn.fit(train_features, train_labels, epochs=1, batch_size=64)

        player.nn.save("RLPlayer.h5")