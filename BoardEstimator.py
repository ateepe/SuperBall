import tensorflow as tf
import BoardGen
import BoardScorer

def CreateNetwork(numRows=8, numCols=10, numColors=5):

    board_column = tf.feature_column.numeric_column(key='board', shape=[numColors+1, numRows * numCols], dtype=tf.int8)

    all_columns = [board_column]

    model_dir = "./savestate"

    num_inputs = numRows*numCols*(numColors+1)

    nn = tf.estimator.DNNRegressor(
        feature_columns=all_columns,
        hidden_units=(num_inputs, num_inputs),
        model_dir=model_dir,
        label_dimension=1
    )

    return nn


def board_to_tensor(board, colors=['.', 'p', 'b', 'y', 'r', 'g']):
    tensor = []

    for c in range(0, len(colors)):
        piece = []
        for i in range(len(board)):
            if board[i] == colors[c]:
                piece.append(1)
            else:
                piece.append(0)
        
        tensor.append(piece)

    return tensor

# converts a batch of inputs (features) and their expected values (labels) and returns a chunk
# of data in a format the network can use to train itself
def train_input_fn(boards, scores, batch_size=1000, repetitions=10):

    formatted_boards = []

    for b in boards:
        formatted_boards.append(board_to_tensor(b))

    features = {'board': formatted_boards}

    dataset = tf.data.Dataset.from_tensor_slices((features, scores))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(batch_size)
    dataset = dataset.repeat(repetitions)
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

# same as my_train_input_fn above, but the labels are optional
# if labels are included, network makes prediction and compares to known result.
# if labels are not included, network simply makes its prediction without knowning what the correct result should be.
# returns chunk of data network can use to make a bunch of predictions in a large batch
def eval_input_fn(boards, scores, batch_size=1000):

    formatted_boards = []

    for b in boards:
        formatted_boards.append(board_to_tensor(b))

    features = {'board': formatted_boards}

    if scores is None:
	    # No labels, use only features.
	    #inputs = eval_features
	    inputs = features
    else:
		#inputs = (eval_features, eval_labels)
        inputs = (features, scores)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    nn = CreateNetwork()

    for i in range(0, 100):
        batch_boards = []
        batch_scores = []
        for j in range(0, 10000):
            b = list(BoardGen.gen_random_data())
            batch_boards.append(b)
            batch_scores.append(BoardScorer.analyze_board(b)[0])

        nn.train(input_fn=lambda:train_input_fn(batch_boards, batch_scores))
        #nn.evaluate(input_fn=lambda:eval_input_fn(batch_boards, batch_scores))

    
