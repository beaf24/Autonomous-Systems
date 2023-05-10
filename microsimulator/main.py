import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

def pythagoras(obstacleX, obstacleY, rosX, rosY):
    dist_x = (obstacleX - rosX) ** 2
    dist_y = (obstacleY - rosY) ** 2
    return math.sqrt(dist_x + dist_y)


mapW = 100
mapH = 100
realMap = [[0] * mapW for _ in range(mapH)]  # realMap -> 100x100
scannedMap = [[0] * mapW for _ in range(mapH)]  # scannedMap -> 100x100
rosX = 20  # initial LIDAR x position
rosY = 80  # initial LIDAR y position
maxRange = 100  # LIDAR range
maxAngle = 1.5 * math.pi  # LIDAR max angle

angleIncrements = np.linspace(math.pi, maxAngle, 100, False)

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
realMap[rosX][rosY] = 2
scannedMap[rosX][rosY] = 2

for row in realMap:
    print(row)

for angle in angleIncrements:
    x2 = rosX + maxRange * math.cos(angle)
    y2 = rosY - maxRange * math.sin(angle)

    for i in range(0, 50):
        u = i/50
        x = int(x2 * u + rosX * (1 - u))
        y = int(y2 * u + rosY * (1 - u))

        if 0 < x < mapW and 0 < y < mapH:
            angleStr = "Angle_" + str(angle)
            if realMap[x][y] == -1:
                dist = pythagoras(x, y, rosX, rosY)
                print("x: " + str(x))
                print("y: " + str(y))
                print("reamMap[x][y]: " + str(realMap[x][y]))
                scanData.append([angleStr, dist, x, y])
                break



for data in scanData:
    scanX = data[2]
    scanY = data[3]
    scannedMap[scanX][scanY] = 1




fig, ax = plt.subplots()
im = ax.imshow(scannedMap)



ax.set_title("Scanned map")
fig.tight_layout()
plt.show()



