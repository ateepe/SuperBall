import RLPlayer
import pickle
import estimatorModel

if __name__ == "__main__":
    player = RLPlayer.RLPlayer()

    boards = []
    rewards = []

    saved_boards = ['boards1.pickle', 'boards2.pickle', 'boards3.pickle', 'boards4.pickle', 'boards5.pickle', 'boards6.pickle', 'boards7.pickle', 'boards8.pickle', 'boards9.pickle', 'boards10.pickle', 'boards11.pickle', 'boards12.pickle', 'boards13.pickle' ]
    saved_rewards = ['rewards1.pickle', 'rewards2.pickle', 'rewards3.pickle', 'rewards4.pickle', 'rewards5.pickle', 'rewards6.pickle', 'rewards7.pickle', 'rewards8.pickle', 'rewards9.pickle', 'rewards10.pickle', 'rewards11.pickle', 'rewards12.pickle', 'rewards13.pickle' ]

    for board_file in saved_boards:
        boards = boards + pickle.load(open(board_file, "rb"))

    for reward_file in saved_rewards:
        rewards = rewards + pickle.load(open(reward_file, "rb"))

    print("num boards:", len(boards))
    print("num rewards:", len(rewards))

    pickle.dump(boards, open("all_boards.pickle", "wb"))
    pickle.dump(rewards, open("all_rewards.pickle", "wb"))
    

    start_features = estimatorModel.split_into_channels([b[0] for b in boards])
    end_features = estimatorModel.split_into_channels([b[1] for b in boards])

    for round in range(0, 25):
        labels = []

        start_scores = player.nn.predict(start_features).flatten()
        end_scores = player.nn.predict(end_features).flatten()

        for i in range(len(boards)):
            start_board = boards[i][0]
            end_board = boards[i][1]

            reward = rewards[i]

            #start_score = player.get_board_score(start_board)
            #end_score = player.get_board_score(end_board)
            start_score = start_scores[i]
            end_score = end_scores[i]

            if reward < 0:
                target = -1
            else:
                target = start_score + player.alpha * ((reward + player.discount_factor * end_score) - start_score)
            
            labels.append(target)

            #print("converging", start_score, "to", target)

        player.nn.fit(start_features, labels, epochs=1, batch_size=128)
        player.nn.save("RLPlayer.h5")