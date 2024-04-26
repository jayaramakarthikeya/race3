#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from math import atan
from ackermann_msgs.msg import AckermannDriveStamped
# TODO CHECK: include needed ROS msg type headers and libraries
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped , TwistStamped

from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import tkinter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('/media/karthik/DATA/sem2/f1tenth/lab1_ws/src/race3/raceline_opt/dissertation-master/python')

import message_filters

from joblib import dump, load

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        # TODO: create ROS subscribers and publishers
        egocar_odom = message_filters.Subscriber(self,Odometry,'/ego_racecar/odom')
        waypoint_sub = message_filters.Subscriber(self,PointStamped,'/waypoint')
        speed_sub = message_filters.Subscriber(self,TwistStamped,'/waypoint_speed')

        ts = message_filters.ApproximateTimeSynchronizer([egocar_odom, waypoint_sub,speed_sub], queue_size=10, slop=0.5)
        ts.registerCallback(self.odom_sub_callback)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_sub_callback, 10)
        self.opp_odom_sub = self.create_subscription(Odometry,'/opp_racecar/odom',self.opp_odom_sub_callback,10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        #self.waypoint_pub = self.create_publisher(PointStamped, '/waypoint', 10)
        self.opp_drive_pub = self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)


        # parameters
        position_lookahead_time = 0.5
        cofs = 0.3
        map = 1
        self.steer_cap = 0.4
        self.steer_gain = 1.0
        self.low_speed_compensation = 0.2

        with open('/media/karthik/DATA/sem2/f1tenth/lab1_ws/src/race3/raceline_opt/dissertation-master/data/plots/my_track/curvature/1_0.6.pickle', 'rb') as f:
            trajectory = pickle.load(f)

        trajectory_position = trajectory.path.position(trajectory.s).T
        trajectory_time = np.diff(trajectory.s) / trajectory.velocity.v
        cumulative_time = np.cumsum(trajectory_time)
        cumulative_time = np.insert(cumulative_time, 0, 0)
        lookahead_s = np.empty(trajectory.s.shape)
        for i, t in enumerate(cumulative_time):
            j = np.searchsorted(cumulative_time, (t + 0.35)%cumulative_time[-1])
            lookahead_s[i] = trajectory.s[j]
            lookahead_trajectory_position = trajectory.path.position(lookahead_s).T

        trajectory_position = trajectory_position[:-1]
        lookahead_trajectory_position = lookahead_trajectory_position[:-1]
        #print(lookahead_trajectory_position.shape,lookahead_trajectory_position[0])
        target = np.hstack((lookahead_trajectory_position, trajectory.velocity.v.reshape(-1,1)))

        self.kn_regressor_op = KNeighborsRegressor()
        self.kn_regressor_op.fit(trajectory_position, target)
        print("PurePursuit Initialized")


    def odom_sub_callback(self, odom_msg:Odometry, waypoint_msg: PointStamped,speed_msg: TwistStamped):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        
        current_position = np.array([[odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y]])
        lookahead_position = [waypoint_msg.point.x,waypoint_msg.point.y]
        speed = speed_msg.twist.linear.x

        self.get_logger().info('Current Position: %f, %f' % (current_position[0][0], current_position[0][1]))

        # TODO: transform goal point to vehicle frame of reference
        arb = R.from_quat([
            odom_msg.pose.pose.orientation.x, 
            odom_msg.pose.pose.orientation.y, 
            odom_msg.pose.pose.orientation.z, 
            odom_msg.pose.pose.orientation.w])
        lookahead_position_in_map_frame = np.insert(lookahead_position-current_position, 2, 0)
        lookahead_position_in_vehicle_frame = arb.inv().apply(lookahead_position_in_map_frame)

        # TODO: calculate curvature/steering angle and speed
        L = np.linalg.norm(lookahead_position_in_vehicle_frame)
        gamma = lookahead_position_in_vehicle_frame[1]/(L**2)
        steering_angle = self.steer_gain * gamma
        # steering_angle = self.steer_gain * atan2(lookahead_position_in_vehicle_frame[1], lookahead_position_in_vehicle_frame[0])
        steering_angle = min(max(steering_angle, -self.steer_cap), self.steer_cap)

        # TODO: publish drive message, don't forget to limit the steering angle.
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        #self.get_logger().info('Publishing: steering_angle: %f, speed: %f' % (steering_angle, speed))
        self.drive_pub.publish(drive_msg)

        # publish the waypoint
        # waypoint_msg = PointStamped()
        # waypoint_msg.point.x = lookahead_position[0][0]
        # waypoint_msg.point.y = lookahead_position[0][1]
        # waypoint_msg.header.frame_id = "map"
        # self.waypoint_pub.publish(waypoint_msg)

    def opp_odom_sub_callback(self,odom_msg:Odometry):
        current_position = np.array([[odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y]])
        target = self.kn_regressor_op.predict(current_position)
        lookahead_position = target[:,0:2]

        self.get_logger().info('Current Position: %f, %f' % (current_position[0][0], current_position[0][1]))

        # TODO: transform goal point to vehicle frame of reference
        arb = R.from_quat([
            odom_msg.pose.pose.orientation.x, 
            odom_msg.pose.pose.orientation.y, 
            odom_msg.pose.pose.orientation.z, 
            odom_msg.pose.pose.orientation.w])
        lookahead_position_in_map_frame = np.insert(lookahead_position-current_position, 2, 0)
        lookahead_position_in_vehicle_frame = arb.inv().apply(lookahead_position_in_map_frame)

        # TODO: calculate curvature/steering angle and speed
        L = np.linalg.norm(lookahead_position_in_vehicle_frame)
        gamma = lookahead_position_in_vehicle_frame[1]/(L**2)
        steering_angle = self.steer_gain * gamma
        # steering_angle = self.steer_gain * atan2(lookahead_position_in_vehicle_frame[1], lookahead_position_in_vehicle_frame[0])
        steering_angle = min(max(steering_angle, -self.steer_cap), self.steer_cap)

        speed = target[0][2]

        # TODO: publish drive message, don't forget to limit the steering angle.
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        #self.get_logger().info('Publishing: steering_angle: %f, speed: %f' % (steering_angle, speed))
        self.opp_drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()