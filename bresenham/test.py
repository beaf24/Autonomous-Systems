import bresAlgo
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

meters_pixel_ratio = 0.01

reading = 0.15
robot_orientation = -math.pi/3
laser_angle = -math.pi/3

mapW = 50
mapH = 50
realMap = [[0] * mapW for _ in range(mapH)]

coords = bresAlgo.determine_coords( x0=14, y0= 15, z_angle=robot_orientation, laser_angle=laser_angle, laser_measure=reading, meters_pixel_ratio=meters_pixel_ratio)

calculate_points = bresAlgo.bres_algo(x0= 14, y0= 15, x1= coords[0], y1= coords[1])

print(coords)
print(calculate_points)

for points in calculate_points:
    pointX = points[0]
    pointY = points[1]
    realMap[pointX][pointY] = 15

realMap[14][15] = 10
realMap[coords[0]][coords[1]] = 20

fig, ax = plt.subplots()
im = ax.imshow(np.transpose(realMap), origin='lower')

ax.set_title("Real map")
fig.tight_layout()
plt.show()
plt.close()
