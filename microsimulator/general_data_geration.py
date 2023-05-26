import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from bresenham import determine_coords
import copy
import os
from mapping_microsim import Static_Map

def pythagoras(obstacleX, obstacleY, rosX, rosY):
    dist_x = (obstacleX - rosX) ** 2
    dist_y = (obstacleY - rosY) ** 2
    return math.sqrt(dist_x + dist_y)

def import_imageMap(filename: str):
    im = Image.open(os.getcwd() + "/microsimulator/" + filename)
    image = np.asarray(im)

    map = copy.deepcopy(image)

    color = np.unique(image)
    print(color)
    true_color = [0, 1, 2]

    for i, c in enumerate(color):
        map[image == c] = true_color[i]
    
    mapW, mapH = map.shape

    fig, ax = plt.subplots()
    im = ax.imshow(map, origin='lower', cmap = 'gray')
    fig.colorbar(im)

    ax.set_title("Scanned map")
    fig.tight_layout()
    plt.show()
    plt.close()

    return list(map), mapW, mapH

def create_realMap(mapW:int = 100, mapH:int = 100):
    realMap = [[0] * mapW for _ in range(mapH)]  # realMap -> 100x100

    # Create walls with thickness of 2 in realMap
    for i in range(0, 100):
        realMap[i][0] = 1
        realMap[i][1] = 1
        realMap[i][99] = 1
        realMap[i][98] = 1

    realMap[0] = [1] * mapW
    realMap[1] = [1] * mapW
    realMap[98] = [1] * mapW
    realMap[99] = [1] * mapW

    return realMap, mapW, mapH

def scan_data(map:list, rosX:int = 80, rosY:int = 80, maxRange:float = 100, minAngle:float = 0, maxAngle = 0.5 * math.pi, increments: int = 100):
    mapW, mapH = np.array(map).shape
    angleIncrements = np.linspace(minAngle, maxAngle, increments, False)
    scanData = []

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
                if map[x][y] == 1:  # Wall found
                    dist = pythagoras(x, y, rosX, rosY)
                    #print("x: " + str(x))
                    #print("y: " + str(y))
                    scanData.append([rosX, rosY, 0, dist, angle])  # Store data
                    break

    array_data = np.asarray(scanData)

    return scanData, array_data, rosX, rosY

def scannedMap(map:np.array, scanData, rosX:int, rosY:int):
    mapW, mapH = np.array(map).shape
    scannedMap = [[0] * mapW for _ in range(mapH)]  # scannedMap -> 100x100
    # Add scanned obstacles positions to scannedMap

    for data in scanData:
        data_X, data_Y = determine_coords(data[0], data[1], data[2], data[4], data[3], 1)
        scanX = data[0]
        scanY = data[1]
        scannedMap[data_X][data_Y] = 1

    scannedMap[rosX][rosY] = 2

    # Plot the map
    fig, ax = plt.subplots()
    im = ax.imshow(np.transpose(scannedMap), origin='lower')

    ax.set_title("Scanned map")
    fig.tight_layout()
    plt.show()
    plt.close()


realMap, mapW, mapH = import_imageMap("piso5 (2).png")
#realMap, mapW, mapH = create_realMap()
scanData, array_data, rosX, rosY = scan_data(realMap, maxAngle=2*np.pi, increments=200, rosX=50, rosY=100)
np.savetxt("scanData.csv", array_data, delimiter=",")
scannedMap(realMap, scanData, rosX, rosY)

static_map = Static_Map(grid_size = mapH)
static_map.mapping_microsimulation(scanData)

