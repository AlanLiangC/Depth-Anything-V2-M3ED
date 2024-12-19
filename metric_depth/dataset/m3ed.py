import cv2
import h5py
from ouster.sdk import client

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import Resize, NormalizeImage, PrepareForNet
import numpy as np

def transform_inv(T):
    T_inv = np.eye(4)
    T_inv[:3,:3] = T[:3,:3].T
    T_inv[:3,3] = -1.0 * ( T_inv[:3,:3] @ T[:3,3] )
    return T_inv

def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom

class M3ED(Dataset):
    def __init__(self, sequence_path, mode, size=(1280, 800)):

        self.mode = mode
        self.size = size
        
        self.include_m3ed_sequence_data(sequence_path)
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def include_m3ed_sequence_data(self, sequence_path):

        f = h5py.File(sequence_path)
        self.sequence_data = f['/ouster/data']
        self.lidar_calib = f['/ouster/calib']
        ouster_metadata_str = f['/ouster/metadata'][...].tolist()
        self.metadata = client.SensorInfo(ouster_metadata_str)
        self.xyzlut = client.XYZLut(self.metadata)

        self.imgs = f['ovc/rgb/data']
        self.img_map_left = f['/ovc/ts_map_prophesee_left_t']
        self.lidar_map_left = f['/ouster/ts_start_map_prophesee_left_t']
        rgb_camera_calib = f['/ovc/rgb/calib']

        Event_T_Lidar = self.lidar_calib['T_to_prophesee_left'][:]
        Event_T_RGB = rgb_camera_calib['T_to_prophesee_left'][:]

        RGB_T_Lidar = transform_inv(Event_T_RGB) @ (Event_T_Lidar)
        self.extristric = RGB_T_Lidar
        K = np.eye(3)
        K[0, 0] = rgb_camera_calib['intrinsics'][0]
        K[1, 1] = rgb_camera_calib['intrinsics'][1]
        K[0, 2] = rgb_camera_calib['intrinsics'][2]
        K[1, 2] = rgb_camera_calib['intrinsics'][3]
        width, height = rgb_camera_calib['resolution']
        self.image_shape = (width, height)
        self.K = K
        self.D = rgb_camera_calib['distortion_coeffs'][:]

    def __len__(self):
        return self.sequence_data.shape[0]

    def get_image(self, idx):

        argmin_index = np.argmin(np.abs(self.img_map_left - self.lidar_map_left[idx]))
        return self.imgs[argmin_index][:,:,::-1] # RGB

    def get_lidar(self, idx):
        ouster_sweep = self.sequence_data[idx][...]
        packets = client.Packets([client.LidarPacket(opacket, self.metadata) for opacket in ouster_sweep], self.metadata)
        scans = iter(client.Scans(packets))
        scan = next(scans)
        xyz = self.xyzlut(scan.field(client.ChanField.RANGE))
        signal = scan.field(client.ChanField.REFLECTIVITY)
        signal = client.destagger(self.metadata, signal)
        signal = np.divide(signal, np.amax(signal), dtype=np.float32)
        points = np.concatenate([xyz, signal[:,:,None]], axis=-1)
        points = points.reshape(-1,4)
        filtered_points = points[~np.all(points[:,:3] == [0, 0, 0], axis=1)]
        return filtered_points

    def get_depth(self, idx):
        points = self.get_lidar(idx)[:,:3]
        points_cam = cart_to_hom(points) @ self.extristric.T
        rvecs = np.zeros((3,1))
        tvecs = np.zeros((3,1))

        pts_img, _ = cv2.projectPoints(points_cam[:,:3].astype(np.float32), rvecs, tvecs,
                self.K, self.D)
        imgpts = pts_img[:,0,:]
        valid_points = (imgpts[:, 1] >= 0) & (imgpts[:, 1] < self.image_shape[1]) & \
                    (imgpts[:, 0] >= 0) & (imgpts[:, 0] < self.image_shape[0]) & \
                    (points_cam[:,2] > 0)
        imgpts = imgpts[valid_points]
        imgpts = imgpts.astype(np.int64)
        depth = points_cam[:,2][valid_points]
        depth_image = np.zeros(self.image_shape) # W x H
        depth_image[imgpts[:, 0], imgpts[:, 1]] = (depth / 90)
        depth_image = np.clip(depth_image, None, 1)
        return depth_image

    def __getitem__(self, item):
        
        image = self.get_image(item) / 255.0
        depth = self.get_depth(item)
        depth = np.transpose(depth)

        sample = self.transform({'image': image, 'depth': depth})
        
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['depth'] = sample['depth']
        
        sample['valid_mask'] = sample['depth'] > 0
                
        return sample


if __name__ == "__main__":

    dataset = M3ED(sequence_path='/home/alan/AlanLiang/Dataset/M3ED/processed/Car/Urban_Day/car_urban_day_city_hall/car_urban_day_city_hall_data.h5',
                   mode='train')
    iteration = iter(dataset)
    sample = iteration.__next__()
    print(sample)