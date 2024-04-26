
import numpy as np
import copy
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, PointStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import message_filters
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsRegressor

class Overtake(Node):
    def __init__(self):
        super().__init__('overtake_node')
        waypoint_topic = "/waypoint_viz"
        self.L = 2.5
        self.time_in_overtake = 0
        self.time_in_return = 0
        self.offsets = np.array([0.7, 0.75, 0.8, 0.85, 0.9])
        self.overtake_sign = -1.0
        self.position_lookahead_time = 0.5

        with open('/media/karthik/DATA/sem2/f1tenth/lab1_ws/src/race3/raceline_opt/dissertation-master/data/plots/my_track/compromise/1_1.0.pickle', 'rb') as f:
            trajectory = pickle.load(f)
    
            trajectory_position = trajectory.path.position(trajectory.s).T
            trajectory_time = np.diff(trajectory.s) / trajectory.velocity.v
            cumulative_time = np.cumsum(trajectory_time)
            cumulative_time = np.insert(cumulative_time, 0, 0)
            lookahead_s = np.empty(trajectory.s.shape)
            for i, t in enumerate(cumulative_time):
                j = np.searchsorted(cumulative_time, (t + self.position_lookahead_time)%cumulative_time[-1])
                lookahead_s[i] = trajectory.s[j]
                lookahead_trajectory_position = trajectory.path.position(lookahead_s).T

            trajectory_position = trajectory_position[:-1]
            lookahead_trajectory_position = lookahead_trajectory_position[:-1]
            print(lookahead_trajectory_position.shape,lookahead_trajectory_position[0])
            plt.figure()
            plt.scatter(x=lookahead_trajectory_position[:,0],y=lookahead_trajectory_position[:,1])
            plt.savefig('traj.png')
            target = np.hstack((lookahead_trajectory_position, trajectory.velocity.v.reshape(-1,1)))

            self.kn_regressor = KNeighborsRegressor()
            self.kn_regressor.fit(trajectory_position, target)
        
        egocar_odom = message_filters.Subscriber(self,Odometry,'/ego_racecar/odom')
        scan_msg = message_filters.Subscriber(self,LaserScan,'/scan')
        self.waypoint_pub_ = self.create_publisher(PointStamped,'/waypoint',qos_profile=10)
        self.waypoint_pub_viz = self.create_publisher(Marker, waypoint_topic,qos_profile=10)
        self.speed_pub_ = self.create_publisher(TwistStamped,'/waypoint_speed',qos_profile=10)

        ts = message_filters.ApproximateTimeSynchronizer([egocar_odom, scan_msg], queue_size=10, slop=0.5)
        ts.registerCallback(self.waypoint_generator)
        self.target_point = None

    def visualize_target(self):
        # Publish target waypoint
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 0
        marker.ns = "target_waypoint"
        marker.type = 1
        marker.action = 0
        marker.pose.position.x = self.target_point[0]
        marker.pose.position.y = self.target_point[1]

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        this_scale = 0.2
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        marker.lifetime.nanosec = int(1e8)

        self.waypoint_pub_.publish(marker)


    def check_collision(self, goal, ranges, L):
        if goal is None:
            return True
        heading_angle = np.degrees(np.arctan2(goal[1], goal[0]))
        index = int(heading_angle + 135) * 4
        votes = np.sum(np.array(ranges[index-(4*4):index+(4*4)]) < L)
        return votes > 0
    
    def waypoint_generator(self, odom_msg, scan_msg):
        overtake = False

        current_position = np.array([[odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y]])
        target = self.kn_regressor.predict(current_position)
        lookahead_position = target[:,0:2]

        overtake_right = [
            self.check_collision((lookahead_position[0][0], lookahead_position[0][1] + offset), scan_msg.ranges, self.L)
            for offset in -1.0 * self.offsets
        ]
        overtake_left = [
            self.check_collision((lookahead_position[0][0], lookahead_position[0][1] + offset), scan_msg.ranges, self.L)
            for offset in self.offsets
        ]
        if np.sum(overtake_right) > np.sum(overtake_left):
            self.overtake_sign = 1.0

        for offset in (self.overtake_sign * self.offsets):
            goal_shifted = [lookahead_position[0][0], lookahead_position[0][1] + offset]
            if not self.check_collision(goal_shifted, scan_msg.ranges, self.L):
                overtake = True
                self.time_in_overtake += 1
                self.time_in_return = 0
                #print("overtake!")
                self.L = 2.3
                self.target_point = goal_shifted
                self.visualize_target()
                break      
        if not overtake:
            return  
        
        waypoint_msg = PointStamped()
        waypoint_msg.point.x = self.target_point[0]
        waypoint_msg.point.y = self.target_point[1]
        waypoint_msg.header.frame_id = "map"
        self.waypoint_pub_.publish(waypoint_msg)

        speed_msg = TwistStamped()
        speed_msg.twist.linear.x = target[2]
        speed_msg.header.frame_id = "map"
        self.speed_pub_.publish(speed_msg)
        
    

def main(args=None):
    rclpy.init(args=args)
    print("Overtake Initialized")
    overtake_node = Overtake()
    rclpy.spin(overtake_node)

    overtake_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()