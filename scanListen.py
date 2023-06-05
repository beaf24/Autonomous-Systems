import rospy
from sensor_msgs.msg import LaserScan

class Listener():
    def __init__(self):
        rospy.init_node('scan_listener', anonymous=False)
        rospy.loginfo("Start!")
        rospy.Subscriber('/scan', LaserScan, self.callback)
        self.ranges = []

    def callback(self, msg):
        self.ranges = msg.ranges


if __name__ == '__main__':
    listen = Listener()
    while not rospy.is_shutdown():
        continue
