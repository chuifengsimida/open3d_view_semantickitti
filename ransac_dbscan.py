from os.path import join as join
import numpy as np
import open3d as o3d
import os
import yaml

config = 'semantic-kitti.yaml'
with open(config, encoding='utf-8') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
mx = max(data['color_map'].keys())
color_map = np.zeros((mx+1, 3))
for key in data['color_map']:
    color_map[key] = data['color_map'][key]



           


vis = o3d.visualization.Visualizer()
vis.create_window()

def parse_calibration(filename):
    """ read calibration file with given filename

        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib

def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename

        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses

def fuse_multi_scan(points, pose0, pose):

    # pose = poses[0][idx]

    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    # new_points = hpoints.dot(pose.T)
    new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

    new_points = new_points[:, :3]
    new_coords = new_points - pose0[:3, 3]
    # new_coords = new_coords.dot(pose0[:3, :3])
    new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
    new_coords = np.hstack((new_coords, points[:, 3:]))

    return new_coords

# 进行ransac和dbscan
def ransac_dbscan(pcd, labels, idx):
    if np.array(pcd.points).shape[0] < 3:
        return labels

    _, inliers  = pcd.segment_plane(distance_threshold=0.129, ransac_n=3, num_iterations=40, probability=0.99999999)
    if idx in inliers:
        labels[inliers] = labels[idx]
    else:
        mask = np.ones((labels.shape[0],))
        mask[inliers] = 0
        mask = mask>0

        db_pcd = o3d.geometry.PointCloud()
        db_pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)[mask])
        

        
        db_labels = np.array(db_pcd.cluster_dbscan(eps=0.1, min_points=5, print_progress=False))
        tmp_labels = np.zeros_like(labels)
        tmp_labels[mask] = db_labels
        mask1 = (tmp_labels==tmp_labels[idx])
        labels[mask1] = labels[idx]
    return labels

def process():

    

    seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    pc_ROOT = 'D:/paper_codes/dataset/SemanticKITTI/sequences'
    label_ROOT = 'D:/paper_codes/dataset/SemanticKITTI/0'


    # multi scans
    scans = 0


    ### x, y
    region = np.array([0.5, 0.5])
    for seq in seqs:
        calib = parse_calibration(join(pc_ROOT, seq, 'calib.txt'))
        poses = parse_poses(join(pc_ROOT, seq, 'poses.txt'), calib)
        frames = os.listdir(join(pc_ROOT, seq, 'velodyne'))
        frames.sort()

        min_frame = int(frames[0][:-4])
        for fidx in range(100, len(frames)):
            pc = np.fromfile(join(pc_ROOT, seq,'velodyne', frames[fidx]), dtype=np.float32).reshape((-1, 4))[:,:3]

            truth = np.fromfile(join(pc_ROOT, seq, 'labels', frames[fidx][:-3]+'label'), dtype=np.int32)
            truth = truth & 0xFFFF

            label = np.load(join(label_ROOT, seq, frames[fidx][:-3]+'npy')).astype(np.uint32)
            cur_point_nums = pc.shape[0]
            
            #####################
            ## fusion
            #####################
            pose0 = poses[fidx]
            for i in range(-scans, scans+1):
                if i==0:
                    continue
                cur_idx = fidx + i
                
                if cur_idx < 0 or cur_idx>=len(poses):
                    break
                pose1 = poses[cur_idx]

                pc1 = np.fromfile(join(pc_ROOT, seq,'velodyne', frames[cur_idx]), dtype=np.float32).reshape((-1, 4))[:,:3]
                label1 = np.load(join(label_ROOT, seq, frames[cur_idx][:-3]+'npy')).astype(np.uint32)
                
                mask1 = label1>0
                pc1, label1 = pc1[mask1], label1[mask1]

                pc1 = fuse_multi_scan(pc1, pose0, pose1)

                pc = np.concatenate((pc, pc1), axis=0)
                label = np.concatenate((label, label1), axis=0)
            #####################
            ## fusion end
            #####################



            ratio_idxs = np.where(label>0)[0]
            for idx in ratio_idxs:
                dis = pc[:,:2] - pc[idx][:2]
 
                mask = np.all(((dis>-region) & (dis<region)), axis=1)
               
                idx1 = np.sum(mask[:idx])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[mask])
                label[mask] = ransac_dbscan(pcd, label[mask], idx1)
            
            pc = pc[:cur_point_nums]
            label = label[:cur_point_nums]

            mask = (label>0)
            print(np.sum(mask), np.sum(truth[mask]==label[mask])/ np.sum(mask))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            color = color_map[label]
            pcd.colors = o3d.utility.Vector3dVector(color)

            vis.add_geometry(pcd)
            vis.run()
            vis.destroy_window()


            return 

if __name__=='__main__':
    process()