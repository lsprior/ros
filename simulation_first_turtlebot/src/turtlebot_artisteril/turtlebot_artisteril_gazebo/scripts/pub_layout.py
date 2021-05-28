#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
import json
import tf2_ros
import math
import numpy as np
import rospkg

if __name__ == '__main__':
    rospy.init_node('pub_layout')

    rate = rospy.Rate(20.0)
    rospy.loginfo("Publishing layout points position...")

    marker_publisher = rospy.Publisher(
        'visualization_marker', Marker, queue_size=2)
    marker_publisher2 = rospy.Publisher(
        'real_time_path', Marker, queue_size=30)

    tfBuffer = tf2_ros.Buffer(rospy.Duration(1200.0))
    listener = tf2_ros.TransformListener(tfBuffer)
    rospy.sleep(5)



    marker_spheres = Marker(
        type=Marker.SPHERE_LIST,
        id=1,
        pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
        scale=Vector3(0.2, 0.2, 0.2),
        header=Header(frame_id='/map'),
        color=ColorRGBA(0, 1, 0.0, 1)
    )

    marker_spheres_stations = Marker(
        type=Marker.SPHERE_LIST,
        id=2,
        pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
        scale=Vector3(0.4, 0.4, 0.4),
        header=Header(frame_id='/map'),
        color=ColorRGBA(1, 1, 0.0, 1)
    )

    marker_lines = Marker(
        type=Marker.LINE_LIST,
        id=3,
        pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
        scale=Vector3(0.1, 0.1, 0.1),
        header=Header(frame_id='/map'),
        color=ColorRGBA(0, 0, 1, 1)
    )

    # marker_curves=Marker(
    #     type=Marker.LINE_LIST,
    #     id=4,
    #     pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
    #     scale=Vector3(0.1, 0.1, 0.1),
    #     header=Header(frame_id='/map'),
    #     color=ColorRGBA(1, 0.7, 0, 1)
    # )

    # layout_list = []

    LOOKUP_TABLE = np.array([
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    87178291200.0,
    1307674368000.0,
    20922789888000.0,
    355687428096000.0,
    6402373705728000.0,
    121645100408832000.0,
    2432902008176640000.0,
    51090942171709440000.0,
    1124000727777607680000.0,
    25852016738884976640000.0,
    620448401733239439360000.0,
    15511210043330985984000000.0,
    403291461126605635584000000.0,
    10888869450418352160768000000.0,
    304888344611713860501504000000.0,
    8841761993739701954543616000000.0,
    265252859812191058636308480000000.0,
    8222838654177922817725562880000000.0,
    263130836933693530167218012160000000.0
     ])

    def Ni(n, i):
        a1=LOOKUP_TABLE[n]
        a2=LOOKUP_TABLE[i]
        a3=LOOKUP_TABLE[n-i]
        ni=a1/(a2*a3)
        return ni

    def Bernstein(n, i, t):
        if t == 0.0 and i == 0:
            ti = 1.0
        else:
            ti = t**i

        if (n == i and t == 1.0):
            tni = 1.0
        else:
            tni = (1-t)**(n-i)

        basis = Ni(n, i)*ti*tni
        return basis

    def Bezier2D(trajectory, segment):
        cpts = segment['Parts']
        npts = len(trajectory)
        interpolatedPoints = [0]*cpts
        icount = 0
        t = 0.0
        step = 1.0/(cpts-1)

        for i in range(cpts):
            if (1.0-t) < 5e-6:
                t = 1.0

            jcount = 0
            interpolatedPoints[icount] = Point(0.0, 0.0, 0.0)

            for j in range(npts):
                basis = Bernstein(npts - 1, j, t)
                interpolatedPoints[icount].x += basis*trajectory[jcount].x
                interpolatedPoints[icount].y += basis*trajectory[jcount].y
                jcount += 1

            icount += 1
            t += step
        return interpolatedPoints



    ############## Read Layout.json and draw points in Rviz ##################################################
    rospack = rospkg.RosPack()
    # with open("/home/rafael/Documents/ROS/WORKSPACES/catkin_ws/src/simulation00/layouts/layout.json") as f:
    # TODO: Real time way to read layout.json provided by C#
    # FIXME: Some error you should work it up

    try:
        with open(rospack.get_path('turtlebot_artisteril_gazebo')+"/maps/batcave_test1.json") as f:
            data = json.load(f)

        trajectory = []

        for station in data['Stations']:
            stationPoint=Point(station['Vector']['X']/1000.0, station['Vector']['Y']/1000.0, 0)
            marker_spheres_stations.points.append(stationPoint)

        for segment in data['Segments']:
            # List of layout points is not according to id
            # layout_list.append([segment['StartVector']['X']/1000, segment['StartVector']['Y']/1000])
            pointStart=Point(segment['StartVector']['X']/1000.0, segment['StartVector']['Y']/1000.0, 0)
            pointEnd=Point(segment['EndVector']['X']/1000.0, segment['EndVector']['Y']/1000.0, 0)
            marker_lines.points.append(pointStart)
            marker_spheres.points.append(pointStart)
            if segment['Vectors']:
                # marker_curves.color()
                trajectory.append(pointStart)
                for ptinvectors in segment['Vectors']:
                    trajectory.append(Point(ptinvectors['X']/1000.0, ptinvectors['Y']/1000.0, 0))
                trajectory.append(pointEnd)
                curve_points=Bezier2D(trajectory,segment)
                # print(curve_points)
                marker_lines.points.extend(curve_points)
                del trajectory[:]
            marker_lines.points.append(pointEnd)
            marker_spheres.points.append(pointEnd)

        # layout = np.asarray(layout_list)
        marker_publisher.publish(marker_lines)
        marker_publisher.publish(marker_spheres)
        marker_publisher.publish(marker_spheres_stations)
    except:
        pass


##############################################################################################

############## Draw odom ###########################
    marker_spheres2 = Marker(
        type=Marker.SPHERE_LIST,
        id=5,
        pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
        scale=Vector3(0.07, 0.07, 0.07),
        header=Header(frame_id='/map'),
        color=ColorRGBA(1, 1, 0.0, 1)
    )

    print("i am here.")
    while not rospy.is_shutdown():
        try:
            transform_msg_direction = tfBuffer.lookup_transform(
                "map", "base_link", rospy.Time(0), rospy.Duration(0.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue

        if transform_msg_direction:
            agv_x = transform_msg_direction.transform.translation.x
            agv_y = transform_msg_direction.transform.translation.y

            marker_spheres2.points.append(Point(agv_x, agv_y, 0))
            marker_publisher2.publish(marker_spheres2)
        rate.sleep()
##############################################################################################
