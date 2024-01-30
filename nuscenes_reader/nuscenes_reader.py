import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
from os.path import join
from std_msgs.msg import String
from nuscenes.nuscenes import NuScenes
import numpy as np
import pypcd as pcd

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('nuscenes_reader')
        self.nusc = NuScenes(version='v1.0-mini',
                             dataroot='/data/sets/nuscenes', verbose=True)
        self.declare_parameter('scene', 0)
        scene_index = self.get_parameter(
            'scene').get_parameter_value().integer_value
        self.scene = self.nusc.scene[scene_index]
        self.cam_token = self.nusc.get(
            'sample', self.scene['first_sample_token'])
        self.radar_token = self.nusc.get(
            'sample', self.scene['first_sample_token'])
        self.lidar_token = self.nusc.get(
            'sample', self.scene['first_sample_token'])
        self.camera_list = ['CAM_FRONT', 'CAM_FRONT_LEFT',
                            'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.radar_list = ['RADAR_FRONT', 'RADAR_FRONT_LEFT',
                           'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        self.lidar_list = ['LIDAR_TOP']

        camera_publisher_names = [
            f'~/camera_{cam.lower().replace("cam_", "")}/image_raw' for cam in self.camera_list]
        self.cam_publishers = [self.create_publisher(
            Image, camera, 1) for camera in camera_publisher_names]
        radar_publisher_names = [
            f'~/radar_{radar.lower().replace("radar_", "")}/pointcloud' for radar in self.radar_list]
        self.radar_publishers = [self.create_publisher(
            PointCloud2, radar, 1) for radar in radar_publisher_names]
        lidar_publisher_names = [
            f'~/lidar_{lidar.lower().replace("lidar_", "")}/pointcloud' for lidar in self.lidar_list]
        self.lidar_publishers = [self.create_publisher(
            Image, lidar, 1) for lidar in lidar_publisher_names]
        # LIDAR 20Hz capture frequency
        # RADAR 13Hz capture frequency
        # camera 12Hz capture frequency
        camera_period = 1/12.0
        radar_period = 1/13.0
        lidar_period = 1/20.0
        self.camera_trigger = self.create_timer(
            camera_period, self.camera_callback)
        self.radar_trigger = self.create_timer(
            radar_period, self.radar_callback)
        self.lidar_trigger = self.create_timer(
            lidar_period, self.lidar_callback)

    def next_token(self, token):
        if token['next'] != '':
            return self.nusc.get('sample', token['next'])
        else:
            return None

    def get_filepath(self, token, sensors):
        data = [self.nusc.get('sample_data', token['data'][sensor])
                for sensor in sensors]
        return [join(self.nusc.dataroot, d['filename']) for d in data]

    def camera_callback(self):
        # get sample data
        # do loop for all cameras in the list
        if self.cam_token is None:
            return
        # rclpy info
        data = self.get_filepath(self.cam_token, self.camera_list)
        # open image
        images = [cv2.imread(d) for d in data]
        # convert to ROS message
        img_msg = [CvBridge().cv2_to_imgmsg(
            image, encoding="bgr8") for image in images]
        # publish
        self.cam_token = self.next_token(self.cam_token)
        for msg, pub in zip(img_msg, self.cam_publishers):
            pub.publish(msg)

    def radar_callback(self):
        # get sample data
        self.get_filepath(self.radar_token, self.radar_list)
        return

    def lidar_callback(self):
        # get sample data
        self.get_filepath(self.lidar_token, self.lidar_list)
        pass


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
