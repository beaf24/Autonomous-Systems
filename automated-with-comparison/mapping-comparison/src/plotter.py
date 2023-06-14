import numpy as np
import matplotlib.pyplot as plt

size_f = open('size.txt', 'r')

size_info = size_f.readlines()[0].split(" ")  # max_x | min_x | max_y | min_y

init_x = np.abs(int(size_info[1])) + 1
init_y = np.abs(int(size_info[3])) + 1

max_x_size = int(size_info[0]) + np.abs(int(size_info[1])) + int(init_x) 
max_y_size = int(size_info[2]) + np.abs(int(size_info[3])) + int(init_y) 


size_f.close()


mapp = [[0] * (max_x_size) for _ in range(max_y_size)]

data_f = open('export.txt', 'r')

data = data_f.readlines() # Each line: cell_x | cell_y | cell_log_odds

data_f.close()


i = 0

for cell in data:
    cell_data = cell.split(" ")

    x = int(cell_data[0]) + init_x
    y = int(cell_data[1]) + init_y

    logOdd = float(cell_data[2])

    mapp[x][y] = mapp[x][y] + logOdd




prob_map = 1 - 1/(1+np.exp(mapp))


fig, ax = plt.subplots()


im = ax.matshow(prob_map, cmap="Greys")
plt.colorbar(im)

plt.show()


