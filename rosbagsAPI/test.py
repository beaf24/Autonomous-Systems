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

class Static_Map:
    def __init__(self, grid_size: int =60, resolution:int = 1):
        self.l0 = np.log(0.5/0.5)
        self.l_free = 0.2 
        self.l_occ = 0.8
        self.grid_size = grid_size # tamanho em metros da grelha
        self.resolution = resolution # resolução das celulas da grelha
        self.logodds = self.l0*np.ones((grid_size*resolution, grid_size*resolution)) #grelha de 60 por 60

    def bresenham(self, start, end):
        """
        Bresenham's Line Generation Algorithm
        https://www.youtube.com/watch?v=76gp2IAazV4
        """
        # step 1 get end-points of line 
        (x0, y0) = start
        (x1, y1) = end

        print(x0, y0, x1, y1)

        # step 2 calculate difference
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        m = dy/dx
        
        # step 3 perform test to check if pk < 0
        flag = True
        
        line_pixel = []
        line_pixel.append((x0,y0))
        
        res = self.grid_size * self.resolution

        step = self.resolution
        if x0>x1 or y0>y1:
            step = -self.resolution

        mm = False   
        if m < 1:
            x0, x1 ,y0 ,y1 = y0, y1, x0, x1
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            mm = True
            
        p0 = 2*dx - dy
        x = x0
        y = y0
        
        for i in range(int(abs(y1-y0))):
            if flag:
                x_previous = x0
                p_previous = p0
                p = p0
                flag = False
            else:
                x_previous = x
                p_previous = p
                
            if p >= 0:
                x = x + step

            p = p_previous + 2*dx -2*dy*(abs(x-x_previous))
            y = y + 1
            
            if mm:
                line_pixel.append((y,x))
            else:
                line_pixel.append((x,y))
                
        line_pixel = np.array(line_pixel).astype(int)
        
        return line_pixel

    def occupancy_grid_mapping(self, x_t:tuple, z_t:np.array):
        res = self.grid_size * self.resolution
        print(z_t)
        
        distance, angle = z_t
        beam_xy = (distance*np.cos(angle), distance*np.sin(angle))
        print(x_t, beam_xy)
        line = self.bresenham(start = x_t, end = beam_xy)
        print(line)
        for mi in line:
            print(mi)
            pos_x, pos_y = mi
            if (mi == line[-1]).all() and distance != 'NaN': # tem de atingir alvo
                self.logodds[pos_x, pos_y] = self.logodds[pos_x, pos_y] + self.l_occ - self.l0
            else:
                self.logodds[pos_x, pos_y] = self.logodds[pos_x, pos_y] + self.l_free - self.l0

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
        im = Image.fromarray(prob_map)
        im.show()

    @staticmethod
    def mapping_microsimulation(scanData:np.array):
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
            angle, dist, x, y = read
            print(angle)
            static_map.occupancy_grid_mapping((x, y), (angle, dist))
        
        return static_map.get_map()

if __name__ == "__main__":
    scanData = np.loadtxt(fname='/Users/Beatriz/Documents/GitHub/Autonomous-Systems/rosbagsAPI/scanData.csv', delimiter=',')
    static_map = Static_Map()
    static_map.mapping_microsimulation(scanData)

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

    



