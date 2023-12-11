from turtle import color, update
import numpy as np
import open3d as o3d
import os
import yaml
from pickle import load, dump
from os.path import join as join

ROOT = r'D:\paper_codes\dataset\SemanticPoss\sequences'
SAVE_ROOT = r'D:\paper_codes\dataset\SemanticPoss\Objects'
seq = '02'
config_file = 'semantic-kitti.yaml'
with open(config_file, encoding='utf-8') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
print('debug')
scans = os.listdir(join(ROOT, seq, 'velodyne'))
for scan in scans:
    pc = np.fromfile(join(ROOT, seq, 'velodyne', scan), dtype=np.float32).reshape(-1, 4)
    label = np.fromfile(join(ROOT, seq, 'labels', scan[:-3]+'label'), dtype=np.int32)
    label = label & 0xFFFF
    print(scan)
    
    for key in config['labels'].keys():
        mask = label == key
        if np.sum(mask)==0:
            continue
        save_path = join(SAVE_ROOT, seq, scan[:-4])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        tmp_pc = pc[mask]
        tmp_pc = tmp_pc.astype(np.float32).copy()
        tmp_pc.tofile(join(save_path, config['labels'][key]+'.bin'))

        


