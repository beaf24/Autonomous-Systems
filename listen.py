import tf2_ros
import rospy
from std_msgs.msg import String


if __name__ == '__main__':
    rospy.init_node('tf2_listener', anonymous=False)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    rospy.wait_for_message(topic= '/tf' , topic_type = String, timeout=10)
    while not rospy.is_shutdown():
        try:
            msg = tfBuffer.lookup_transform(target_frame='map', source_frame='base_link', time=rospy.Time())
            pos = msg.transform.translation
            ori = msg.transform.rotation
            print(pos)
            print(ori)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue