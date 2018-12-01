import SuperBall
import numpy as np
import estimatorModel

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import Sequential, load_model

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

        try:
            self.nn = load_model("RLPlayer.h5")
        except:
            self.nn = estimatorModel.define_model()


    def generate_episode(self):
        self.game.StartGame()

        while not self.game.gameOver:

            starting_board = np.copy(self.game.board)

            reward = 0

            if self.game.numOpenTiles < 5:
                pass
                #attempt to score

            else:
                swap, intermediate_board = self.choose_swap(self.game.board)
                
                (r1, c1), (r2, c2) = swap

                if self.game.swap(r1, c1, r2, c2):
                    reward = 0
                else:
                    reward = -1
        
            final_board = np.copy(self.game.board)

            starting_score = self.get_board_score(starting_board)
            intermediate_score = self.get_board_score(intermediate_board)
            final_score = self.get_board_score(final_board)

            # TODO: add to training data and train after episode over

    # returns swapped indices and resulting board
    #  ( (index1, index2), board )
    def choose_swap(self, board):
        
        swaps = []
        potential_boards = []

        for index1 in range(0, self.game.numTiles):
            for index2 in range(index1+1, self.game.numTiles):
                if board[index1] != '.' and board[index2] != '.':
                    r1 = int(index1 // self.game.numCols)
                    c1 = int(index1 % self.game.numCols)
                    r2 = int(index2 // self.game.numCols)
                    c2 = int(index2 % self.game.numCols)
                    
                    swapped_board = np.copy(board)
                    swapped_board[index1] = board[index2]
                    swapped_board[index2] = board[index1]

                    swaps.append( ((r1, c1), (r2, c2)) )
                    potential_boards.append(swapped_board)

        predictions = self.nn.predict(potential_boards).flatten()

        best_index = predictions.argmax()

        return (swaps[best_index], potential_boards[best_index])

    # returns scored tile and resulting board
    #  ( (row, col), board )
    def choose_score(self, board):
        # TODO: choose which set to score
        pass
    
    def get_board_score(self, board):
        features = estimatorModel.split_into_channels([board])
        score = self.nn.predict(features).flatten()[0]
        return score

    # TODO: train network on observed data
    def train_network(self):
        pass

if __name__ == "__main__":
    pass