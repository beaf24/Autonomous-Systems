import rospy
from sensor_msgs.msg import LaserScan
import os

class Listener():
    def __init__(self):
        rospy.init_node('scan_listener', anonymous=False)
        rospy.Subscriber('/scan', LaserScan, self.callback)
        self.msg = []

    def callback(self, msgread):
        self.msg = msgread


if __name__ == '__main__':
    listen = Listener()
    while rospy.wait_for_message("/scan", LaserScan, timeout=10):
        rospy.loginfo(listen.msg)
