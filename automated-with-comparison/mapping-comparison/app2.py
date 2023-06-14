import subprocess
import os
import numpy as np


try:
    os.remove("./data/bag_data.txt")
except:
    print("file does not exist, creating it")




## -------------------------------------------------------------------------------------------------------------------##
# 1st SUBPROCESS - START
# Read and export meaningful data from rosbag


bag_file = input("Type rosbag file name: ")  # type bag file name

p = subprocess.Popen('python3 ./src/export_data.py -file /bags/' + bag_file, stdout=subprocess.PIPE, shell=True)
out, err = p.communicate()
output = out.decode("utf-8")
print(output)
if "error" in output: exit(1)
else: print("Data exported to: bag_data.txt")


# 1st SUBPROCESS - END
## -------------------------------------------------------------------------------------------------------------------##


########################################################################################################################


## -------------------------------------------------------------------------------------------------------------------##
# 2nd SUBPROCESS - START
# Read and process exported data 


ratio = input("Type meter/pixel ratio: ")  # type meter/pixel ratio
# log_odd_free = input("Indicate probability odds for free cell ]0 ; 0.5[ : ")  # type log odds for free cell

# while( float(log_odd_free) > 0.5 or float(log_odd_free) < 0 ):
#     print("Probability odds must be ]0 ; 0.5[")
#     log_odd_free = input("Indicate probability odds for free cell ]0 ; 0.5[ : ")


for p_free in np.arange(0.10, 0.50, 0.01):
    try:
        os.remove("./data/export.txt")
    except:
        print("file does not exist")

    p_free = np.format_float_positional(p_free, precision=2)    

    p = subprocess.Popen('./src/mapping ' + ratio + " " + str(p_free), stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    output = out.decode("utf-8")
    print(output)
    if "ERROR" in output: exit(1)
    else: print("Processed data exported to: export.txt")


# 2nd SUBPROCESS - END
## -------------------------------------------------------------------------------------------------------------------##


########################################################################################################################


## -------------------------------------------------------------------------------------------------------------------##

# 3rd SUBPROCESS - START
# Plot processed data and save as .png


    p = subprocess.Popen('python3 ./src/plotter2.py -p_free ' + str(p_free), stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()


# 3rd SUBPROCESS - END
## -------------------------------------------------------------------------------------------------------------------##

# p = subprocess.Popen('mogrify -rotate -90 ./images/map_algo*.png', stdout=subprocess.PIPE, shell=True)
# out, err = p.communicate()
# output = out.decode("utf-8")
# print(output)

for p_free in np.arange(0.10, 0.50, 0.01):
    p_free = np.format_float_positional(p_free, precision=2)
    p = subprocess.Popen('python3 ./comparison/image_comparison.py -p_free ' + str(p_free) + " -res " + ratio + " -name " + bag_file, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
