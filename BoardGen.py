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

def save_npz(path='datasets/randomBoards.npz', num_training_samples=60000, num_testing_samples=10000):
	"""Save randomly generated boards and scores as training/testing data.

	Generates boards randomly using the gen_random_data() function (make
	the data generation function modular?) and then saves the generated
	boards in a .npz file representing training and testing data for
	neural networks.

	Arguments:
		path (str): The path to the .npz file to save.
		num_training_samples (int): The number of training samples to generate.
		num_testing_samples (int): The numebr of testing samples to generate.
	"""

	x_train = []
	y_train = []
	for n in range(num_training_samples):
		data = gen_random_data()
		score, score_pos = bs.analyze_board(data)
		x_train.append(data)
		y_train.append(score)

	x_test = []
	y_test = []
	for n in range(num_testing_samples):
		data = gen_random_data()
		score, score_pos = bs.analyze_board(data)
		x_test.append(data)
		y_test.append(score)

	# Convert lists to numpy arrays
	np.array(x_train)
	np.array(y_train)
	np.array(x_test)
	np.array(y_test)

	# Save the arrays to a .npz file
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
	board = gen_large_set_data()
	score, score_pos = bs.analyze_board(board)
	print_data(board)
	print(score)
	exit()

	# Test the board generation/save/load
	save_npz(num_training_samples=2, num_testing_samples=2)
	(x_train, y_train), (x_test, y_test) = load_data()
	print('x_train')
	print(x_train)
	print('y_train')
	print(y_train)
	print('x_test')
	print(x_test)
	print('y_test')
	print(y_test)
