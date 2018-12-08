import SuperBall
import BoardScorer
import numpy as np
import pickle
import copy

if __name__ == "__main__":
    print("run 13")
    player = SuperBall.SuperBall()

    boards = []
    rewards = []

    for i in range(0, 75):
        print("episode", i)

        player.StartGame()

        while not player.gameOver:
            start_board = copy.deepcopy(player.board)

            if player.numOpenTiles < 5:
                score, score_pos = BoardScorer.analyze_board(start_board)

                if score_pos == (-1, -1):
                    boards.append((start_board, start_board))
                    rewards.append(-1)

                    player.gameOver = True
                    print("game over")
            
                else:
                    tiles_scored = player.score_ignore_color(score_pos[0], score_pos[1])

                    end_board = copy.deepcopy(player.board)
                    boards.append((start_board, end_board))
                    rewards.append(tiles_scored)

                    #print("score", score_pos, "reward =", tiles_scored)
                    print("total score:", player.totalScore)
            
            else:
                swaps = []
                potential_boards = []

                for index1 in range(0, player.numTiles):
                    for index2 in range(index1+1, player.numTiles):
                        if player.board[index1] != '.' and player.board[index2] != '.':
                            r1 = int(index1 // player.numCols)
                            c1 = int(index1 % player.numCols)
                            r2 = int(index2 // player.numCols)
                            c2 = int(index2 % player.numCols)
                    
                            swapped_board = np.copy(player.board)
                            swapped_board[index1] = player.board[index2]
                            swapped_board[index2] = player.board[index1]

                            swaps.append( ((r1, c1), (r2, c2)) )
                            potential_boards.append(swapped_board)

                predictions = np.array([BoardScorer.analyze_board(b)[0] for b in potential_boards])

                best_index = predictions.argmax()

                best_swap = swaps[best_index]

                (r1, c1), (r2, c2) = best_swap

                player.swap(r1, c1, r2, c2)

                end_board = copy.deepcopy(player.board)

                reward = 0

                boards.append((start_board, end_board))
                rewards.append(reward)

                #print("swap", best_swap, "reward = ", reward)

    pickle.dump(boards, open("boards13.pickle", "wb"))
    pickle.dump(rewards, open("rewards13.pickle", "wb"))