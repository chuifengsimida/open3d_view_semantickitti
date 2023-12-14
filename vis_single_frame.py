import open3d as o3d
import yaml
import numpy as np
from os.path import join 

config = 'waymo.yaml'

with open(config, encoding='utf-8') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
mx = max(data['color_map'].keys())
color_map = np.zeros((mx+1, 3))
for key in data['color_map']:
    color_map[key] = data['color_map'][key]
color_map = color_map / 255.0

vis = o3d.visualization.Visualizer()
vis.create_window()
render_option = vis.get_render_option()
render_option.point_size = 1
render_option.background_color = np.asarray([255, 255, 255])


############
# add pcd
#############
ROOT = r'C:\Users\87770\Desktop'
pc = np.loadtxt(join(ROOT, '1550083469745187_pc.txt')).reshape(-1, 3)
labels = np.loadtxt(join(ROOT, '1550083469745187_label.txt'))[:,1].astype(np.int32)

color_map[0] = [255, 255, 255]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color_map[labels])
vis.add_geometry(pcd)


vis.run()
vis.destroy_window()

