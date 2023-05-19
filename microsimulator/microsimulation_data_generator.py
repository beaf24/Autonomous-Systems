import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from bresenham import determine_coords

def pythagoras(obstacleX, obstacleY, rosX, rosY):
    dist_x = (obstacleX - rosX) ** 2
    dist_y = (obstacleY - rosY) ** 2
    return math.sqrt(dist_x + dist_y)


mapW = 100
mapH = 100
realMap = [[0] * mapW for _ in range(mapH)]  # realMap -> 100x100
scannedMap = [[0] * mapW for _ in range(mapH)]  # scannedMap -> 100x100
rosX = 80  # initial LIDAR x position
rosY = 80  # initial LIDAR y position
maxRange = 100  # LIDAR range
maxAngle = 0.5 * math.pi  # LIDAR max angle

angleIncrements = np.linspace(0, maxAngle, 100, False)

scanData = []

# Create walls with thickness of 2 in realMap
for i in range(0, 100):
    realMap[i][0] = -1
    realMap[i][1] = -1
    realMap[i][99] = -1
    realMap[i][98] = -1

realMap[0] = [-1] * mapW
realMap[1] = [-1] * mapW
realMap[98] = [-1] * mapW
realMap[99] = [-1] * mapW

# Set robot position
realMap[rosX][rosY] = 2
scannedMap[rosX][rosY] = 2

# For each laser beam
for angle in angleIncrements:
    x2 = rosX + maxRange * math.cos(angle)
    y2 = rosY + maxRange * math.sin(angle)
    
    # Divide the beam by 50 parts (this number can be tweaked...) and iterate through each part 
    # from the origin untill the obstacle/end
    for i in range(0, 50):
        u = i/50
        # Interpolation
        x = int(rosX + (x2 - rosX) * u)    # note that rosX and rosY are the inital LiDAR position
        y = int(rosY + (y2 - rosY) * u)

        if 0 < x < mapW and 0 < y < mapH:
            angleStr = "Angle_" + str(angle)
            if realMap[x][y] == -1:  # Wall found
                dist = pythagoras(x, y, rosX, rosY)
                #print("x: " + str(x))
                #print("y: " + str(y))
                scanData.append([rosX, rosY, 0, dist, angle])  # Store data
                break

array_data = np.asarray(scanData)
np.savetxt("scanData.csv", array_data, delimiter=",")

# Add scanned obstacles positions to scannedMap

for data in scanData:
    data_X, data_Y = determine_coords(data[0], data[1], data[2], data[4], data[3], 1)
    scanX = data[0]
    scanY = data[1]
    scannedMap[data_X][data_Y] = 1

# Plot the map
fig, ax = plt.subplots()
im = ax.imshow(np.transpose(scannedMap), origin='lower')

ax.set_title("Scanned map")
fig.tight_layout()
plt.show()
plt.close()