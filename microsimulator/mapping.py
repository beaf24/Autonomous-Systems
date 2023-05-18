import rosbag
import numpy as np
import matplotlib.pyplot as plt
from quaternion_to_euler import z_quaternion_to_euler
from laserData import LaserData
from bresenham import determine_coords, bres_algo

max_map_x = 1000
max_map_y = 1000

init_x = max_map_x/2 - 1
init_y = max_map_y/2 - 1

robot_x = max_map_x/2 - 1
robot_y = max_map_y/2 - 1
meter_per_pixel = 0.05

map = [[0] * max_map_x for _ in range(max_map_y)]

robot_orientation = 0

bag = rosbag.Bag("piso5_4_amcl.bag")


for topic, msg, t in bag.read_messages(topics=['/amcl_pose', '/scan']):
    if (topic == "/amcl_pose"):
        relative_pos = msg.pose.pose.position  # in meters, we still need to convert to the equivalent in pixels
        relative_orientation = msg.pose.pose.orientation
        robot_x = init_x + (relative_pos.x / meter_per_pixel)
        robot_y = init_y + (relative_pos.y / meter_per_pixel)
        robot_orientation = z_quaternion_to_euler( w= relative_orientation.w, z= relative_orientation.z)

    if(topic == "/scan"):
        laserScan = LaserData( minAngle= msg.angle_min, maxAngle= msg.angle_max, angleIncrement= msg.angle_increment, ranges= msg.ranges)

        for measure in laserScan.laserData:  # attribute "laserData" in "laserScan" is a list of laser beams, with each having a distance measured and angle of the measurement
            if( str( measure.distance ) == "nan") : continue  

            obstacle_x, obstacle_y = determine_coords( x0= robot_x, y0= robot_y, z_angle= robot_orientation, laser_angle= measure.angle, laser_measure= measure.distance, meters_pixel_ratio= meter_per_pixel)
            
            free_cells = bres_algo(robot_x, robot_y, obstacle_x, obstacle_y)

            for cell in free_cells:
                map[cell[0]][cell[1]] = map[cell[0]][cell[1]] + 0.2

            map[obstacle_x][obstacle_y] = map[obstacle_x][obstacle_y] + 0.8


for row in map:
    for cell in row:
        cell = 1 - 1/(1+np.exp(cell))
        if cell > 0.8:
            cell = 10
        elif cell < 0.2:
            cell = 5
        else:
            cell = 0
        


# Plot the map
fig, ax = plt.subplots()
im = ax.imshow(map, origin='upper')

ax.set_title("Scanned map")
fig.tight_layout()
plt.show()
plt.close()

            

            


        

    

