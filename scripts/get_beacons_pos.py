#!/usr/bin/env python
####
# Check the mean beam of the lidar and use it to make lidar parallel to ground.

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
import json
import tf2_ros
import math
import numpy as np
import rospkg
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray

beacons_found=[]

def callback(msg):
    try:
        for i in range(0,100):
            # print msg.markers[1].pose.position
            if msg.markers[i].ns=="landmarks/world" and msg.markers[i].color.b!=1.0:
                print '{},{},{}'.format(i,msg.markers[i].pose.position.x,msg.markers[i].pose.position.y)
    except :
        pass




if __name__ == '__main__':
    rospy.init_node('beacons_pos')
    rate = rospy.Rate(30.0)
    rospy.Subscriber('rviz_markers', MarkerArray, callback)

while not rospy.is_shutdown():
    rate.sleep()
