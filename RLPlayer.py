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

    def __init__(self, nRows=8, nCols=10, goals=SuperBall.default_goals, colors=SuperBall.default_colors, minSetSize=5):
        self.game = SuperBall.SuperBall(nRows, nCols, goals, colors, minSetSize);
        self.discount_factor = 0.95

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

        #print(self.game.board)

        train_boards = []
        train_labels = []

        while not self.game.gameOver:

            starting_board = np.copy(self.game.board)

            reward = 0

            if self.game.numOpenTiles < 5:
                best_score = self.choose_score(self.game.board)

                # no score possible, end of game
                if best_score is None:
                    reward = -1
                    self.game.gameOver = True
                    intermediate_board = self.game.board
                
                else:
                    (score_row, score_col) = best_score[0]
                    intermediate_board = best_score[1]

                    print("scoring", (score_row, score_col))

                    reward = self.game.score(score_row, score_col)

                    if reward <= 0:
                        print("something messed up, tried to score when we can't")
                        exit(0)

            else:
                swap, intermediate_board = self.choose_swap(self.game.board)
                
                (r1, c1), (r2, c2) = swap

                print("Swapping", (r1, c1), "and", (r2, c2))

                if self.game.swap(r1, c1, r2, c2):
                    reward = 0
                else:
                    reward = -1
        
            final_board = np.copy(self.game.board)

            if self.game.gameOver:
                train_boards.append(starting_board)
                train_labels.append(-1)
            
            else:
                '''
                starting_score = self.get_board_score(starting_board)
                intermediate_score = self.get_board_score(intermediate_board)
                final_score = self.get_board_score(final_board)
                '''
                scores = self.get_board_scores([starting_board, intermediate_board, final_board])
                starting_score = scores[0]
                intermediate_score = scores[1]
                final_score = scores[2]


            train_boards.append(starting_board)
            train_labels.append(reward + self.discount_factor * intermediate_score)

            train_boards.append(intermediate_board)
            train_labels.append(0 + self.discount_factor * final_score)
            
            #print(self.game.board)

        print("end episode with score of", self.game.totalScore)
        return (train_boards, train_labels)

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

        predictions = self.nn.predict(estimatorModel.split_into_channels(potential_boards)).flatten()

        best_index = predictions.argmax()

        return (swaps[best_index], potential_boards[best_index])

    # returns scored tile and resulting board
    #  ( (row, col), board )
    def choose_score(self, board):
        djset = self.game.getDJSet()

        score_tiles = []
        score_boards = []

        for i in range(0, self.game.numTiles):
            if self.game.isGoal(i) and board[i] != '.':
                setID = djset.getSetID(i)
                setSize = djset.getSetSize(i)
                if setSize >= self.game.minSetSize:
                    new_board = np.copy(board)

                    for j in range(0, self.game.numTiles):
                        if djset.getSetID(j) == setID:
                            new_board[j] = '.'

                    r = int(i // self.game.numCols)
                    c = int(i % self.game.numCols)
                    
                    score_tiles.append((r, c))
                    score_boards.append(new_board)

        # find best board after scoring
        if len(score_tiles) > 0:
            features = estimatorModel.split_into_channels(score_boards)
            predictions = self.nn.predict(features).flatten()

            best_index = predictions.argmax()

            return (score_tiles[best_index], score_boards[best_index])
        
        # can't score
        else:
            return None

        pass

    def get_board_scores(self, boards):
        features = estimatorModel.split_into_channels(boards)
        score = self.nn.predict(features).flatten()
        return score

    def get_board_score(self, board):
        features = estimatorModel.split_into_channels([board])
        score = self.nn.predict(features).flatten()[0]
        return score

    # TODO: train network on observed data
    def train_network(self):
        pass

if __name__ == "__main__":
    player = RLPlayer()

    while True:
        train_boards, train_labels = player.generate_episode()

        train_features = estimatorModel.split_into_channels(train_boards)
        
        player.nn.fit(train_features, train_labels, epochs=10, batch_size=len(train_labels))

        player.nn.save("RLPlayer.h5")