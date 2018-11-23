import BoardScorer as bs
import numpy as np

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

# TensorFlow dataset format:
# x_train, y_train, x_test, y_test
# Where x is the chararray board input and y is the score of that board from Alex's algorithm
def save_npz(outfile, num_training_samples=60000, num_testing_samples=10000):
	

	for n in range(num_samples):
		print()
	return


data = gen_random_data()
print_data(data)

if (data[0] == '.'):
	print('data[0] is equivalent to \'.\'')

score, score_pos = bs.analyze_board(data)

print(score)




