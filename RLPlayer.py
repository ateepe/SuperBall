import SuperBall
import numpy as np
import estimatorModel
import BoardScorer

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
        self.alpha = 0.1

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

    def generate_episode(self, bOffline=False):
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

                    #reward = self.game.score(score_row, score_col)
                    reward = self.game.score_ignore_color(score_row, score_col)

                    if reward <= 0:
                        print("something messed up, tried to score when we can't")
                        exit(0)

                    #reward = 1 #override to ignore actual value of scoring set and just train to score more sets

            else:
                swap, intermediate_board = self.choose_swap(self.game.board, bOffline=bOffline)
                
                (r1, c1), (r2, c2) = swap

                #print("Swapping", (r1, c1), "and", (r2, c2))

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


                #reward = 0 # override reward of 0 for all things that are not losing 
                train_boards.append(starting_board)
                corrected_score = starting_score + self.alpha * ((reward + self.discount_factor * final_score) - starting_score)
                train_labels.append(corrected_score)

                print("starting board score: ", starting_score)
                print("corrected board score:", corrected_score)
                #print("shifting board score of ", starting_score, "to", corrected_score)


            #print(self.game.board)

        print("end episode with score of", self.game.totalScore)
        return (train_boards, train_labels)

    # returns swapped indices and resulting board
    #  ( (index1, index2), board )
    def choose_swap(self, board, bOffline=False):
        
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

        if bOffline:
            predictions = np.array([BoardScorer.analyze_board(b)[0] for b in potential_boards])
        else:
            predictions = self.nn.predict(estimatorModel.split_into_channels(potential_boards)).flatten()

        best_index = predictions.argmax()

        '''
        from sortedcontainers import SortedDict

        sd = SortedDict()
        for i in range(len(potential_boards)):
            sd[predictions[i]] = potential_boards[i]
        
        #for val, board in sd.items():
            #print("board value: ", val)
            #self.print_board(board)
        '''
        
        #print("\n\n")
        print("chose index", best_index)
        #print(predictions[best_index])
        #self.print_board(potential_boards[best_index])
        
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

    def print_board(self, board):
        for r in range(self.game.numRows):
            for c in range(self.game.numCols):
                print(board[r*self.game.numCols + c], end="")
            print()

if __name__ == "__main__":
    player = RLPlayer()

    episode_number = 1
    while True:
        train_boards = []
        train_labels = []
        
        # generate a few episodes at a time so it can train on a larger set of data
        for i in range(0, 1):
            if (episode_number % 1) == 0 and False:
                print("offline")
                boards, labels = player.generate_episode(bOffline=True) # offline learn from hard coded algorithm
            else:
                print("online")
                boards, labels = player.generate_episode(bOffline=False) # network learns online by self
            train_boards = train_boards + boards
            train_labels = train_labels + labels
            episode_number = episode_number + 1

        #train_features = estimatorModel.split_into_channels(train_boards)
        
        #player.nn.fit(train_features, train_labels, epochs=1, batch_size=64)

        #player.nn.save("RLPlayer.h5")