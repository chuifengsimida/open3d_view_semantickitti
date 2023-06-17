from turtle import color, update
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import yaml
import utils
from pickle import load, dump
from os.path import join as join



Dataset = ['waymo']
ROOT = r'D:/paper_codes/dataset/Waymo/validation'
config = 'waymo.yaml'
with open(config, encoding='utf-8') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
mx = max(data['color_map'].keys())
color_map = np.zeros((mx+1, 3))
for key in data['color_map']:
    color_map[key] = data['color_map'][key]
color_map = color_map / 255.0
point_size = 3

min_bound = np.array([-75, -75, -2]).astype(np.float64)
max_bound = np.array([75, 75, 4]).astype(np.float64)

BoundingBox = None
BoundingBox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound = min_bound,
    max_bound = max_bound 
)

scans = 20
downsamping = 100000

class AppWindow:

    def __init__(self, width, height):
        
        self.window = gui.Application.instance.create_window(
            "view semantic segmentation", width, height
        )
        w = self.window
        w.renderer.set_clear_color([255,255,255,0.5])

        self.dataset = Dataset[0]
        self.seq = None
        self.frame = 0

        self.poses = None
        self.pcs = None
        self.labels = None
        self.scans = 1
        self.min_frame = None
        self.max_frame = None
        self.downsamping = False
        self.baseline = False
        self.filter_plane = False
        self.dbscan_flag = False

        self.ransac_distance = 0.129
        self.ransac_points = 3
        self.ransac_iters = 40

        self.dbscan_eps = 0.1
        self.dbscan_min_points = 5


        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self._scene = gui.SceneWidget()
        self._scene_rendering = rendering.Open3DScene(w.renderer)
        #self._scene_rendering.set_background([1,1,1,1])
        self._scene_rendering.set_background([1,1,1,10]) # 全白
        self._scene.scene = self._scene_rendering


        self.geometry_render = o3d.visualization.rendering.MaterialRecord()
        self.geometry_render.point_size = point_size
        


        # 数据集
        self._dataset = gui.Combobox()
        for name in Dataset:
            self._dataset.add_item(name)
        self._dataset.set_on_selection_changed(self._on_change_dataset)
        
        self._seq = gui.Combobox()
        self.seq_names = [i for i in os.listdir(ROOT) if os.path.isdir(join(ROOT, i))]
        self.seq_names.sort()
        for name in self.seq_names:
            self._seq.add_item(name)
        self.seq = self.seq_names[0]
        self.seq_id = 0
        self._seq.set_on_selection_changed(self._on_change_seq)

        self._frames = gui.Slider(gui.Slider.INT)
        self._frames.set_on_value_changed(self._on_change_frame)

        label_dirs = [item for item in os.listdir(join(ROOT, self.seq)) if os.path.isdir(join(ROOT, self.seq, item)) and item not in ['velodyne', 'voxel']]

        self._label = gui.Combobox()
        for name in label_dirs:
            self._label.add_item(name)
        self.label_dir = label_dirs[1]
        self._label.set_on_selection_changed(self._on_change_label)
        
        self.pcs = os.listdir(join(ROOT, self.seq, 'velodyne'))
        self.labels = os.listdir(join(ROOT, self.seq, self.label_dir))
        self.poses = np.load(join(ROOT, self.seq, 'poses.npy')) #self.parse_poses(join(ROOT, self.seq, 'poses.txt'), self.calib)
        self.pcs.sort()
        self.labels.sort()
        self.min_frame = int(self.pcs[0][:-4])
        self.max_frame = int(self.pcs[-1][:-4])
        self._frames.set_limits(self.min_frame, self.max_frame)

        self._previouce = gui.Button('previous')
        self._previouce.set_on_clicked(self._on_previous)
        self._next = gui.Button('next')
        self._next.set_on_clicked(self._on_next)

        
        self._scans = gui.Slider(gui.Slider.INT)
        self._scans.set_on_value_changed(self._on_change_scans)
        self._scans.set_limits(int(1), int(scans))

        self.h1 = gui.Horiz(em)
        self._baseline = gui.Checkbox('Baseline')
        self._baseline.set_on_checked(self._on_change_baseline)
        self._downsamping = gui.Checkbox('DownSampling')
        self._downsamping.set_on_checked(self._on_change_downsamping)
        
        self.h1.add_child(self._baseline)
        self.h1.add_child(self._downsamping)

        self.h2 = gui.Horiz(em)
        self._filter_plane = gui.Checkbox('Filter Plane')
        self._filter_plane.set_on_checked(self._on_change_filter_plane)
        self._dbscan = gui.Checkbox("DBSCAN")
        self._dbscan.set_on_checked(self._on_change_dbscan)


        self._ransac_dis = gui.Slider(gui.Slider.DOUBLE)
        self._ransac_dis.set_limits(0.01, 0.5)
        self._ransac_dis.double_value = self.ransac_distance
        self._ransac_dis.set_on_value_changed(self._on_change_ransac_dis)

        self._ransac_points = gui.Slider(gui.Slider.INT)
        self._ransac_points.set_limits(int(3), 100)
        self._ransac_points.int_value = self.ransac_points
        self._ransac_points.set_on_value_changed(self._on_change_ransac_points)

        self._ransac_iters = gui.Slider(gui.Slider.INT)
        self._ransac_iters.set_limits(int(1), int(100))
        self._ransac_iters.int_value = self.ransac_iters
        self._ransac_iters.set_on_value_changed(self._on_change_ransac_iters)

        self._dbscan_eps = gui.Slider(gui.Slider.DOUBLE)
        self._dbscan_eps.set_limits(0.01, 0.5)
        self._dbscan_eps.double_value = self.dbscan_eps
        self._dbscan_eps.set_on_value_changed(self._on_change_dbscan_eps)

        self._dbscan_min_points = gui.Slider(gui.Slider.INT)
        self._dbscan_min_points.set_limits(int(1), int(100))
        self._dbscan_min_points.int_value = self.dbscan_min_points
        self._dbscan_min_points.set_on_value_changed(self._on_change_dbscan_min_points)




        self.h2.add_child(self._filter_plane)
        self.h2.add_child(self._dbscan)
        
        self.h3 = gui.Horiz(em)
        self._save_view = gui.Button('Save View')
        self._save_view.set_on_clicked(self.save_view)
        self._load_view = gui.Button('Load View')
        self._load_view.set_on_clicked(self.load_view)
        self.h3.add_child(self._save_view)
        self.h3.add_child(self._load_view)

        

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self._dataset)
        self._settings_panel.add_child(self._seq)
        self._settings_panel.add_child(self._label)
        self._settings_panel.add_child(gui.Label('Frame:'))
        self._settings_panel.add_child(self._frames)
        self._settings_panel.add_child(self._previouce)
        self._settings_panel.add_child(self._next)
        self._settings_panel.add_child(gui.Label('Multi scans:'))
        self._settings_panel.add_child(self._scans)
        self._settings_panel.add_child(self.h1)
        self._settings_panel.add_child(self.h2)
        self._settings_panel.add_child(gui.Label('RANSAC distance:'))
        self._settings_panel.add_child(self._ransac_dis)
        self._settings_panel.add_child(gui.Label('RANSAC point nums'))
        self._settings_panel.add_child(self._ransac_points)
        self._settings_panel.add_child(gui.Label('RANSAC Iterors'))
        self._settings_panel.add_child(self._ransac_iters)
        self._settings_panel.add_child(gui.Label('DBSCAN eps'))
        self._settings_panel.add_child(self._dbscan_eps)
        self._settings_panel.add_child(gui.Label('DBSCAN min points'))
        self._settings_panel.add_child(self._dbscan_min_points)
        self._settings_panel.add_child(self.h3)

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        
        
        self.update()
        
    
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def _on_change_dataset(self, content, index):
        if content != self.dataset:
            self.dataset = content
            self.update()
    
    def _on_change_label(self, content, index):
        self.label_dir = content
        self.labels = os.listdir(join(ROOT, self.seq, self.label_dir))
        self.labels.sort()
        self.update()

    
    def _on_change_seq(self, content, index):
        self.seq_id = index
        if self._seq.selected_text != content:
            self._seq.selected_text = content
            self._seq.selected_index = index
        if content != self.seq:
            self.seq = content
            self.poses = np.load(os.path.join(ROOT, self.seq, 'poses.npy'))
            
            self.pcs = os.listdir(join(ROOT, self.seq, 'velodyne'))
            if not os.path.exists(join(ROOT, self.seq, self.label_dir)):
                print('{} not found'.format(self.label_dir))
                exit()
            self.labels = os.listdir(join(ROOT, self.seq, self.label_dir))
            self.pcs.sort()
            self.labels.sort()
            self.min_frame = int(self.pcs[0][:-4])
            self.max_frame = int(self.pcs[-1][:-4])
            self._frames.set_limits(self.min_frame, self.max_frame)

            self.frame = 0
            self._frames.int_value = self.frame

            self.update()
    
    def _on_change_frame(self, new_value):
        if new_value < self.min_frame or new_value > self.max_frame:
            if new_value > self.max_frame and self.seq_id+1<len(self.seq_names):
                self._on_change_seq(self.seq_names[self.seq_id+1], self.seq_id+1)
            else:
                return 
            return 

        self.frame = int(new_value)
        self._frames.int_value = self.frame
        print(self.frame)
        self.update()
    
    def _on_change_scans(self, new_value):
        self.scans = int(new_value)
        print("scans: {}".format(self.scans))
        self.update()
    
    def _on_previous(self):
        self._on_change_frame(self.frame - 1)
    
    def _on_next(self):
        self._on_change_frame(self.frame + 1)
    
    def _on_change_downsamping(self, checked):
        if checked:
            self.downsamping = True
        else:
            self.downsamping = False
        self.update()

    def _on_change_baseline(self, checked):
        if checked:
            self.baseline = True
        else:
            self.baseline = False
        self.update()

    def _on_change_filter_plane(self, checked):
        if checked:
            self.filter_plane = True
            self.ransac()
        else:
            self.filter_plane = False
            self.update()
    
    def _on_change_dbscan(self, checked):
        if checked:
            self.dbscan_flag = True
            self.DBSCAN()
        else:
            self.dbscan_flag = False
            self.update()

    def _on_change_ransac_dis(self, new_value):
        self.ransac_distance = new_value
        if self.filter_plane:
            self.ransac()

    def _on_change_ransac_points(self, new_value):
        self.ransac_points = int(new_value)
        if self.filter_plane:
            self.ransac()
    
    def _on_change_ransac_iters(self, new_value):
        self.ransac_iters = int(new_value)
        if self.filter_plane:
            self.ransac()

    def _on_change_dbscan_eps(self, new_value):
        self.dbscan_eps = new_value
        if self.dbscan_flag:
            self.DBSCAN()

    def _on_change_dbscan_min_points(self, new_value):
        self.dbscan_min_points = int(new_value)
        if self.dbscan_flag:
            self.DBSCAN()

    def get_label(self, frame):

        if self.labels[frame].endswith('.label'):
            label = np.fromfile(join(ROOT, self.seq, self.label_dir, self.labels[frame]), dtype=np.int32)
            print(np.max(label), np.min(label))
            instance_label = label//(2**16)
            label = label & 0xFFFF
            

        elif self.labels[frame].endswith('.npy'):
            label = np.load(join(ROOT, self.seq, self.label_dir, self.labels[frame])).astype(np.int32)
            

        return label, instance_label

    def update(self):
        if self.dataset == 'waymo':

            assert(self.frame == int(self.pcs[self.frame].split('/')[-1][:-4]))
            # print(join(ROOT, self.seq, 'velodyne',self.pcs[self.frame]))
            pc = np.fromfile(join(ROOT, self.seq, 'velodyne',self.pcs[self.frame]), dtype=np.float32)
            pc = pc.reshape((-1, 4))
            pc = pc[:,:3]

            #pc = np.load(join(ROOT, self.seq, 'velodyne',self.pcs[self.frame]))
            pose0 = self.poses[self.frame - self.min_frame]


            label, isinstance_label = self.get_label(self.frame)

            
            for i in range(1, self.scans):
                cur_frame = self.frame - i
                if cur_frame < self.min_frame:
                    break
                pose1 = self.poses[cur_frame - self.min_frame]
                pc1 = np.fromfile(join(ROOT, self.seq, 'velodyne',self.pcs[cur_frame]), dtype=np.float32)
                pc1 = pc1.reshape((-1, 4))
                pc1 = pc1[:,:3]

                label1 = self.get_label(cur_frame)


                pc1 = self.transform_point_cloud(pc1, self.poses[cur_frame].reshape(4,4), pose0.reshape(4,4))
                #pc1 = self.fuse_multi_scan(pc1, pose0, pose1)

                if len(pc1) > 0:
                    pc = np.concatenate((pc, pc1), axis=0)

                    if self.baseline:
                        label1 = np.zeros_like(label1)
                    
                    label = np.concatenate((label, label1), axis=0)
     

            #idx = (pc.shape[0]-1) - np.arange(pc.shape[0])
            #pc = pc[idx]
            #label = label[idx]

            # mask = label>0
            # pc, label = pc[mask], label[mask]


            if self.downsamping and pc.shape[0] > downsamping:
                idx = np.random.choice(pc.shape[0], downsamping, replace=False)
                print('idx:', idx.shape)
                pc = pc[idx]
                label = label[idx]
            
            # undefine_label_idx = label > 0
            # pc = pc[undefine_label_idx]
            # label = label[undefine_label_idx]
            
            # mask = np.logical_or(pc[:,2]>4.2, pc[:,2]<-4)
            # pc = pc[mask]
            # label = label[mask]

            # print(np.max(pc, 0))
            # print(np.min(pc, 0))

            mask = label==7

            pc = pc[mask]
            label = label[mask]
            isinstance_label = isinstance_label[mask]
            print(isinstance_label) 

            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(pc)
            color = color_map[isinstance_label%22]
            self.pcd.colors = o3d.utility.Vector3dVector(color)

            if BoundingBox is not None:
                self.pcd = self.pcd.crop(BoundingBox)
            
            print('pc shape: {}'.format(np.asarray(self.pcd.points).shape))

            #print(self.pcs[self.frame], np.max(pc, 0), np.min(pc, 0))

            self._scene.scene.clear_geometry()
            
            self._scene.scene.add_geometry('frame {}'.format(self.frame),self.pcd, self.geometry_render)

            self._filter_plane.checked = False
            self.ransac_pcd = self.pcd


    def ransac(self):
        _, inliers  = self.pcd.segment_plane(distance_threshold=self.ransac_distance, ransac_n=self.ransac_points, num_iterations=self.ransac_iters, probability=0.99999999)
        
        self.ransac_pcd = self.pcd.select_by_index(inliers, invert=True)
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry('frame {}'.format(self.frame),self.ransac_pcd, self.geometry_render)
        


    
    def transform_point_cloud(self, pc, old_pose, new_pose):
        """Transform a point cloud from one vehicle frame to another.

        Args:
            pc: N x d float32 point cloud, where the final three dimensions are the
                cartesian coordinates of the points in the old vehicle frame.
                [x,y,z,...]
            old_pose: 4 x 4 float32 vehicle pose at the timestamp of the point cloud
            new_pose: 4 x 4 float32 vehicle pose to transform the point cloud to.
        """
        # Extract the 3x3 rotation matrices and 3x1 translation vectors from the
        # old and new poses.
        
        # (3, 3)
        old_rot = old_pose[:3, :3]
        # (3, 1)
        old_trans = old_pose[:3, 3:4]
        # (3, 3)
        new_rot = new_pose[:3, :3]
        # (3, 1)
        new_trans = new_pose[:3, 3:4]

        # Extract the local cartesian coordinates from the N x 6 point cloud, adding
        # a new axis at the end to work with np.matmul.
        # (N, 3, 1)
        local_cartesian_coords = pc[..., 0:3][..., np.newaxis]

        # Transform the points from local coordinates to global using the old pose.
        # (N, 3, 1)
        global_coords = old_rot @ local_cartesian_coords + old_trans


        # Transform the points from global coordinates to the new local coordinates
        # using the new pose.
        # (N, 3, 1)
        new_local = np.matrix.transpose(new_rot) @ (global_coords - new_trans)

        # Reassign the dimensions of the range image with the cartesian coordinates
        # in
        new_pc = new_local[..., 0]
        return new_pc
    

    # def fuse_multi_scan(self, points, pose0, pose):

    #     # pose = poses[0][idx]

    #     hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    #     # new_points = hpoints.dot(pose.T)
    #     new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

    #     new_points = new_points[:, :3]
    #     new_coords = new_points - pose0[:3, 3]
    #     # new_coords = new_coords.dot(pose0[:3, :3])
    #     new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
    #     new_coords = np.hstack((new_coords, points[:, 3:]))

    #     return new_coords
    

    def save_view(self):
        vis = self._scene
        fname='saved_view_waymo.pkl'
        try:
            model_matrix = np.asarray(vis.scene.camera.get_model_matrix())
            extrinsic = utils.model_matrix_to_extrinsic_matrix(model_matrix)
            width, height = self.window.size.width, self.window.size.height
            intrinsic = utils.create_camera_intrinsic_from_size(width, height)
            saved_view = dict(extrinsic=extrinsic, intrinsic=intrinsic, width=width, height=height)
            
            with open(fname, 'wb') as pickle_file:
                dump(saved_view, pickle_file)
        except Exception as e:
            print(e)

    def load_view(self):
        vis = self._scene
        fname='saved_view_waymo.pkl'
        try:
            with open(fname, 'rb') as pickle_file:
                saved_view = load(pickle_file)
            
            vis.setup_camera(o3d.camera.PinholeCameraIntrinsic(self.window.size.width, self.window.size.height, saved_view['intrinsic']), saved_view['extrinsic'], BoundingBox)
            # Looks like the ground plane gets messed up, no idea how to fix
        except Exception as e:
            print("Can't find file", e)

    def DBSCAN(self):

        labels = np.array(self.ransac_pcd.cluster_dbscan(eps=self.dbscan_eps, min_points=self.dbscan_min_points))
        
        tmp_pcd = self.ransac_pcd
        color_norm = ((labels-np.min(labels, 0))/(np.max(labels, 0)-np.min(labels, 0))).reshape((-1, 1))
        colors = np.concatenate([color_norm, 1-color_norm, 0.5-color_norm], 1)
 
        tmp_pcd.colors = o3d.utility.Vector3dVector(colors)
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry('frame {}'.format(self.frame),tmp_pcd, self.geometry_render)


def main():
    gui.Application.instance.initialize()
    w = AppWindow(1024, 768)
    try:
        gui.Application.instance.run()
    except:
        exit()

if __name__ == '__main__':
    main()
        