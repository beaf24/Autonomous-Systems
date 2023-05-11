import rosbags
import struct
import pprint
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr

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

# create reader instance
with Reader('parado.bag') as reader:
    # topic and msgtype information is available on .connections list
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # iterate over messages
    for connection, timestamp, rawdata in reader.messages():
        # Deserialize data from topic /scan
        if connection.topic == '/scan':
            msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
            # Add each scan to laserData list
            laserData.append(LaserData(minAngle=msg.angle_min, maxAngle=msg.angle_max, angleIncrement= msg.angle_increment, ranges= msg.ranges))

#for each scan (=LaserData object) iterate for each angle increment (=Range object)
for data in laserData:
    for range in data.laserData:
        print("Angle: " + str(range.angle) + " rads   ||||   Distance: " + str(range.distance) + " meters")
        
    break  # break so it doesnt print too much, just prints the first scan
