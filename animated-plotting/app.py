import subprocess


bag_file = input("Type rosbag file name: ")  # type bag file name

p = subprocess.Popen('python3 export_data.py -file ' + bag_file, stdout=subprocess.PIPE, shell=True)
out, err = p.communicate()
output = out.decode("utf-8")
print(output)
if "error" in output: exit(1)
else: print("\nData exported to: bag_data.txt\n")

ratio = input("Type ratio: ")


p = subprocess.Popen('./mapping ' + ratio, stdout=subprocess.PIPE, shell=True)
out, err = p.communicate()
output = out.decode("utf-8")
if "ERROR" in output: exit(1)
else: print("\nProcessed data exported to: export.txt\n")

print("Plotting...\n")
p = subprocess.Popen('python3 plotter.py', stdout=subprocess.PIPE, shell=True)
out, err = p.communicate()

print(output)
