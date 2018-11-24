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
