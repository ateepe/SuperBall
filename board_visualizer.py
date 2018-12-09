import tkinter as tk
from PIL import Image, ImageTk

# The tile colors for Superball
tiles = ['p', 'b', 'y', 'r', 'g']
goals = \
    [False, False, False, False, False, False, False, False, False, False,\
    False, False, False, False, False, False, False, False, False, False,\
    True, True, False, False, False, False, False, False, True, True,\
    True, True, False, False, False, False, False, False, True, True,\
    True, True, False, False, False, False, False, False, True, True,\
    True, True, False, False, False, False, False, False, True, True,\
    False, False, False, False, False, False, False, False, False, False,\
    False, False, False, False, False, False, False, False, False, False] 

root = tk.Tk()

# Set up the width and height of each tile (in text units)
tile_w = 3
tile_h = 2

filename = 'board_out'
with open(filename, 'r') as f:
	lines = f.readlines()

dimensions = [int(x) for x in lines[0].split(' ')]

print(lines)


for r in range(dimensions[0]):
	for c in range(dimensions[1]):
		
		# Determine the board position
		index = r*dimensions[1] + c

		print('lines:', lines[1][index], 'tiles[0]:', tiles[0])

		# Determine the color of the tile
		if (str(lines[1][index]) == tiles[0]):
			bg = 'purple'
		elif (lines[1][index] == tiles[1]):
			bg = 'lightblue'
		elif (lines[1][index] == tiles[2]):
			bg = 'yellow'
		elif (lines[1][index] == tiles[3]):
			bg = 'red'
		elif (lines[1][index] == tiles[4]):
			bg = 'lightgreen'
		else:
			bg = 'lightgray'

		# If the current index corresponds to a goal cell, add asterisk
		if (goals[index] == True):
			text = '*'
		else:
			text = ''

		# Create the tile
		tk.Label(root, image='', text=text, bg=bg, bd=1, relief='solid', height=tile_h, width=tile_w).grid(row=r, column=c)

# Spin up the GUI
root.mainloop()