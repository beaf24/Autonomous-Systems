import argparse
import rosbag
import os

parser = argparse.ArgumentParser()
parser.add_argument("-file", "--filepath", type=str)
args = parser.parse_args()
bag_file = args.filepath

init_x = 100
init_y = 100

robot_x = init_x
robot_y = init_y
meter_per_pixel = 0.1
try:
    bag = rosbag.Bag(bag_file)
except:
    print("error: bagfile does not exist")
try:
    os.remove("./bag_data.txt")
except:
    print("file does not exist, creating it")

try:
    os.remove("./export2.txt")
    os.remove("./size2.txt")
except:
    print("file does not exist")

file1 = open("bag_data.txt", "+a")

i = 0
export_scan = False

for topic, msg, t in bag.read_messages(topics=['/amcl_pose', '/scan']):
    if (topic == "/amcl_pose"):
        relative_pos = msg.pose.pose.position  # in meters, we still need to convert to the equivalent in pixels
        relative_orientation = msg.pose.pose.orientation
        export_scan = True

    
    if(topic == "/scan" and export_scan):
        inc = 0
        
        if i == 1:
            i = 0
            export_scan = False
            continue
        for measure in msg.ranges:
            if( str(measure) == "nan" or measure > msg.range_max - 0.1 or measure < msg.range_min): 
                inc = inc + 1
                continue
            file1.write( str( f'{relative_pos.x:.10f}' ) + " " + str( f'{relative_pos.y:.10f}' ) + " " + str(relative_orientation.w) + " " + str(relative_orientation.z) + " ")
            file1.write( str( msg.angle_min + (inc * msg.angle_increment) ) + " " + str(measure) + "\n")
            inc = inc + 1

        i = i+1
            

file1.close()

exit(0)
