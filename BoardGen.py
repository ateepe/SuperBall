import DisjointSet as dj
import BoardScorer as bs
import numpy as np

# Used to enumerate the different tile possibilities
tiles = ['.', 'p', 'b', 'y', 'r', 'g']

def gen_random_data():
	new_data = np.chararray((80), unicode=True)
	for i in range(len(new_data)):
		tile_index = np.random.randint(len(tiles))
		new_data[i] = tiles[tile_index]
	return new_data

def gen_large_set_data(max_set_size=5, rows=8, cols=10):
	"""Generate boards with larger tile sets that will score higher.

	Arguments:
		max_set_size (int): The maximum size of neighboring disjoint sets. If
		                    the max size changes randomly (and favors lower
		                    set sizes), the board will be more realistic.
		rows (int): The number of rows on the SuperBall board.
		cols (int): The number of cols on the SuperBall board.

	Returns:
		np.chararray: An array representing the different tile values.
	"""

	board = np.chararray((80), unicode=True)
	djSet = dj.DisjointSet(rows*cols)

	# Determine possible places for unions of sets randomly
	set_divisions = set()
	for r in range(rows-1):
		for c in range(cols):
			cell_index = r*cols + c
			coin_flip = np.random.binomial(1, 0.5, 1)[0]
			if (coin_flip == 1):
				set_divisions.add(cell_index)
	for r in range(rows):
		for c in range(cols-1):
			cell_index = (r*cols + c) + (rows*cols)
			coin_flip = np.random.binomial(1, 0.5, 1)[0]
			if (coin_flip == 1):
				set_divisions.add(cell_index)

	# Union the sets if both neighbors are less than max_set_size
	# TODO --> Change max_set_size randomly to get more realistic boards
	# TODO --> Probably favor lower set sizes with max_set_size
	for s in set_divisions:
		if (s < rows*cols):
			size1 = djSet.getSetSize(s)
			size2 = djSet.getSetSize(s + cols)
			if (size1 < max_set_size and size2 < max_set_size):
				djSet.union(s, s+cols)
		else:
			size1 = djSet.getSetSize(s - rows*cols)
			size2 = djSet.getSetSize(s - rows*cols + 1)
			if (size1 < max_set_size and size2 < max_set_size):
				djSet.union(s - rows*cols, s - rows*cols + 1)

	# Choose a color for each disjoint set and fill in board
	set_colors = {}
	for r in range(rows):
		for c in range(cols):
			index = r*cols + c
			setID = djSet.getSetID(index)
			if setID not in set_colors:
				tile_index = np.random.randint(len(tiles))
				set_colors[setID] = tiles[tile_index]
			board[index] = set_colors[setID]

	return board


def print_data(data):
	for r in range(8):
		for c in range(10):
			print(data[r*10+c], end=' ')
		print()

def generate_dataset(gen_function=gen_random_data, num_training_samples=60000, num_testing_samples=10000):
	"""Return generated boards and scores as training/testing data.

	Arguments:
		gen_function (function name): The name of the function used to
		                              generate a dataset.
		num_training_samples (int): The number of training samples to generate.
		num_testing_samples (int): The numebr of testing samples to generate.
	Returns:
		(tuple), (tuple): Numpy arrays with the training and test data.
	"""
	x_train = []
	y_train = []
	for n in range(num_training_samples):
		data = gen_function()
		score, score_pos = bs.analyze_board(data)
		x_train.append(data)
		y_train.append(score)

	x_test = []
	y_test = []
	for n in range(num_testing_samples):
		data = gen_function()
		score, score_pos = bs.analyze_board(data)
		x_test.append(data)
		y_test.append(score)

	# Convert lists to numpy arrays
	np.array(x_train)
	np.array(y_train)
	np.array(x_test)
	np.array(y_test)

	return (x_train, y_train), (x_test, y_test)

def save_npz(x_train, y_train, x_test, y_test, path='datasets/randomBoards.npz'):
	"""Save randomly generated boards and scores as training/testing data.

	Saves a dataset of training and testing boards/scores to a .npz file
	given by the location in the path argument.

	Arguments:
		path (str): The path to the .npz file to save.
		x_train (numpy array): Training boards.
		y_train (numpy array): Training board scores.
		x_test (numpy array): Testing boards.
		y_test (numpy array): Testing board scores.
	"""
	np.savez(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
	return

def load_data(path='datasets/randomBoards.npz'):
	"""Loads a dataset saved as a .npz file.

	Loads training and testing data. The data is stored such that x_train and
	x_test are arrays of board configurations and y_train and y_test are arrays
	of scores given by Alex's algorithm.

	Arguments:
		path (str): The path to the .npz file to load.
	Returns:
		(tuple), (tuple): Numpy arrays with the training and test data.
	"""

	with np.load(path) as f:
		x_train, y_train = f['x_train'], f['y_train']
		x_test, y_test = f['x_test'], f['y_test']
	return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
	# Code to test the larger sets board generator
	# TODO --> Question: does this generator behave like the completely random one
	#          when the max_set_size is set to 1?
	
	# board = gen_large_set_data()
	# score, score_pos = bs.analyze_board(board)
	# print_data(board)
	# print(score)
	# exit()

	# Test the board generation/save/load

	# THIS STUFF IS OUT OF DATE AFTER UPDATING THE BOARD GENERATION FUNCTION

	# save_npz(gen_function=gen_large_set_data, num_training_samples=1000, num_testing_samples=2)
	# (x_train, y_train), (x_test, y_test) = load_data()
	# print('x_train:', x_train.shape)
	# print('y_train:', y_train.shape)
	# print('x_test:', x_test.shape)
	# print('y_test:', y_test.shape)
	pass
