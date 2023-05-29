import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge     

size_f = open('size2.txt', 'r')

size_info = size_f.readlines()[0].split(" ")  # max_x | min_x | max_y | min_y

init_x = np.abs(int(size_info[1])) + 1
init_y = np.abs(int(size_info[3])) + 1

max_x_size = int(size_info[0]) + np.abs(int(size_info[1])) + int(init_x) 
max_y_size = int(size_info[2]) + np.abs(int(size_info[3])) + int(init_y) 

max_laser_angle = 119.9
min_laser_angle = -135
max_laser_range = 5.6 / float(size_info[4].strip())




size_f.close()


mapp = [[0] * (max_x_size) for _ in range(max_y_size)]

data_f = open('export2.txt', 'r')

data = data_f.readlines() # Each line: cell_x | cell_y | cell_log_odds

data_f.close()


i = 0
j = 0
posX = 0
posY = 0
plt.ion() # enable real-time plotting
plt.figure(1) # create a plot
for cell in data:
    
    cell_data = cell.split(" ")
    
    x = int(cell_data[2]) + init_x
    y = int(cell_data[3]) + init_y
    

    logOdd = float(cell_data[4])
    robot_orient = float(cell_data[5].strip()) * (180/np.pi)
    

    plt.clf()

    mapp[x][y] = mapp[x][y] + logOdd

    if(logOdd > 0 and i < 5):
            if j > 3:
                  j = 0
            else:
                  j = j + 1
            i = i + 1
            circle = plt.Circle((posY + init_y, posX + init_x), radius=1.0, color="r")
            larser_radius = Wedge( (posY + init_y, posX + init_x) , max_laser_range, min_laser_angle + robot_orient - 90, max_laser_angle + robot_orient - 90, fc="b", alpha=0.3) 
            ax = plt.gca()
            ax.add_patch(circle)
            ax.add_artist(larser_radius)
            plt.imshow(1 - 1/(1+np.exp(mapp)), cmap='Greys')
            plt.pause(0.005)
    else:
            if(posX != int(cell_data[0]) or posY != int(cell_data[1])):
                posX = int(cell_data[0])
                posY = int(cell_data[1])
                i = 0
                







# prob_map = 1 - 1/(1+np.exp(mapp))


# fig, ax = plt.subplots()


# im = ax.matshow(prob_map, cmap="Greys")
# plt.colorbar(im)

# plt.show()


