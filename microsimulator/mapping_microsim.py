import rosbags
import struct
import pprint
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
from PIL import Image
import sys
import csv
from sys import stdin
import matplotlib.pyplot as plt
from bresenham import determine_coords, bres_algo

class Range:
    def __init__(self, distance, angle):
        self.distance = distance
        self.angle = angle


class LaserData:
    laserData = []
    def __init__(self, minAngle, maxAngle, angleIncrement, ranges):
        self.minAngle = minAngle
        self.maxAngle = maxAngle
        self.angleIncrement = angleIncrement
        i = 0

        for range in ranges:
            ang = i*angleIncrement + minAngle

            new_range = Range(distance= range, angle= ang)
            self.laserData.append(new_range)
            i += 1

laserData = []

class Static_Map():
    def __init__(self, grid_size: int =100, resolution:int = 1):
        self.l0 = np.log(0.5/0.5)
        
        self.P_free = 0.2 
        self.P_occ = 0.8
        self.grid_size = 100 # tamanho em metros da grelha
        self.resolution = resolution # resolução das celulas da grelha
        self.logodds = np.zeros((100, 100), dtype=float) #grelha de 60 por 60

    def occupancy_grid_mapping(self, x_t:tuple, z_t:tuple):
        x_robot, y_robot = x_t
        x_obstacle, y_obstacle = z_t
        
        bres_algo(x_robot, y_robot, x_obstacle, y_obstacle, self.logodds, np.log(self.P_free/self.P_occ))

        self.logodds[x_obstacle][y_obstacle] = self.logodds[x_obstacle][y_obstacle] + np.log(self.P_occ/self.P_free)
        # print("line: ")
        # print(line)
        # for mi in line:
        #     print(mi)
        #     pos_x, pos_y = mi
        #     if mi == line[-1]: # tem de atingir alvo
        #         self.logodds[pos_x, pos_y] = self.logodds[pos_x, pos_y] + self.l_occ - self.l0
        #     else:
        #         self.logodds[pos_x, pos_y] = self.logodds[pos_x, pos_y] + self.l_free - self.l0

    def logodds_to_prob(self):
        """
        Converter logodds em probabilidades
        """
        res = self.grid_size * self.resolution
        map_prob = np.zeros(self.logodds.shape)
        for i in np.arange(res):
            for j in np.arange(res):
                map_prob[i, j] = 1 - 1/(1+np.exp(self.logodds[i, j]))
        return map_prob
    
    def get_map(self):
        prob_map = self.logodds_to_prob()

        fig, ax = plt.subplots()
        im = ax.imshow(prob_map, origin='upper')

        ax.set_title("Scanned map - Microsimulation")
        fig.tight_layout()
        plt.show()

    def mapping_microsimulation(self, scanData:np.array):
        # with open(filename) as csv_file:
        #     reader = csv.reader(csv_file)
        #     for line in reader:
        #         angle, dist, x, y = line
        #         static_map.occupancy_grid_mapping((x, y), (angle, dist))

        # data = sys.stdin.readlines()
        # for line in csv.reader(data):
        #     angle, dist, x, y = line
        #     static_map.occupancy_grid_mapping((x, y), (angle, dist))

        for read in scanData:
            robot_x, robot_y, robot_angle, laser_dist, laser_angle = read
            # angle_dect, dist_dect, x, y, angle = read
            if laser_dist != 'NaN':
                z_t = determine_coords(robot_x, robot_y, robot_angle, laser_angle, laser_dist, self.resolution)
                rosX, rosY = z_t
                self.occupancy_grid_mapping((100-robot_x-1, robot_y), (100-rosX-1, rosY))
        
        return self.get_map()

if __name__ == "__main__":
    scanData = np.loadtxt(fname='/Users/Beatriz/Documents/GitHub/Autonomous-Systems/microsimulator/scanData.csv', delimiter=',')
    static_map = Static_Map(scanData)
    static_map.mapping_microsimulation(scanData)
    

## READ DATA FROM ROS TOPICS com rosbags
    # # create reader instance
    # with Reader('05_05_Scan&Pose/parado.bag') as reader:
    #     # topic and msgtype information is available on .connections list
    #     for connection in reader.connections:
    #         print(connection.topic, connection.msgtype)

    #     # iterate over messages
    #     for connection, timestamp, rawdata in reader.messages():
    #         # Deserialize data from topic /scan
    #         if connection.topic == '/scan':
    #             msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
    #             # Add each scan to laserData list
    #             laserData.append(LaserData(minAngle=msg.angle_min, maxAngle=msg.angle_max, angleIncrement= msg.angle_increment, ranges= msg.ranges))

    # #for each scan (=LaserData object) iterate for each angle increment (=Range object)
    # for data in laserData:
    #     for range in data.laserData:
    #         print("Angle: " + str(range.angle) + " rads   ||||   Distance: " + str(range.distance) + " meters")
    
    #     break  # break so it doesnt print too much, just prints the first scan

    



