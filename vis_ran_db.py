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



Dataset = ['SemanticPoss', 'SemanticKITTI']
ROOT = r'E:\paper_codes\dataset\SemanticKITTI\sequences'
config = 'semantic-kitti.yaml'
with open(config, encoding='utf-8') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
mx = max(data['color_map'].keys())
color_map = np.zeros((mx+1, 3))
for key in data['color_map']:
    color_map[key] = data['color_map'][key]
color_map = color_map / 255.0
point_size = 2
min_bound = np.array([50, 50, 5]).astype(np.float64)
BoundingBox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound = -min_bound,
    max_bound = min_bound 
)

# lower_bound = np.array([10,0,-5]).astype(np.float64)
# area = np.array([20,20,10]).astype(np.float64)

# BoundingBox = o3d.geometry.AxisAlignedBoundingBox(
#     min_bound = lower_bound,
#     max_bound = lower_bound + area 
# ) #None
scans = 100
downsamping = 100000
#print(data['color_map'])


class AppWindow:

    def __init__(self, width, height):
        
        self.window = gui.Application.instance.create_window(
            "view semantic segmentation", width, height
        )
        w = self.window

        self.dataset = Dataset[1]
        self.seq = None
        self.frame = 0

        self.calib = None
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
        self._scene_rendering.set_background(np.array([255,255,255,255])/255.0)
        self._scene.scene = self._scene_rendering


        self.geometry_render = o3d.visualization.rendering.MaterialRecord()
        self.geometry_render.point_size = point_size
        


        # 数据集
        self._dataset = gui.Combobox()
        for name in Dataset:
            self._dataset.add_item(name)
        self._dataset.set_on_selection_changed(self._on_change_dataset)
        
        self._seq = gui.Combobox()
        names = [i for i in os.listdir(ROOT) if os.path.isdir(join(ROOT, i))]
        names.sort()
        for name in names:
            self._seq.add_item(name)
        self.seq = names[0]
        self._seq.set_on_selection_changed(self._on_change_seq)

        self._frames = gui.Slider(gui.Slider.INT)
        self._frames.set_on_value_changed(self._on_change_frame)
        self.calib = self.parse_calibration(join(ROOT, self.seq, 'calib.txt'))
        self.poses = self.parse_poses(join(ROOT, self.seq, 'poses.txt'), self.calib)
        self.pcs = os.listdir(join(ROOT, self.seq, 'velodyne'))
        self.labels = os.listdir(join(ROOT, self.seq, 'labels'))
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
        self._update = gui.Button('Update')
        self._update.set_on_clicked(self.update)
        self.h3.add_child(self._save_view)
        self.h3.add_child(self._load_view)
        self.h3.add_child(self._update)

        

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self._dataset)
        self._settings_panel.add_child(self._seq)
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
    
    def _on_change_seq(self, content, index):
        if content != self.seq:
            self.seq = content
            self.calib = self.parse_calibration(join(ROOT, self.seq, 'calib.txt'))
            self.poses = self.parse_poses(join(ROOT, self.seq, 'poses.txt'), self.calib)
            self.pcs = os.listdir(join(ROOT, self.seq, 'velodyne'))
            self.labels = os.listdir(join(ROOT, self.seq, 'labels'))
            self.pcs.sort()
            self.labels.sort()
            self.min_frame = int(self.pcs[0][:-4])
            self.max_frame = int(self.pcs[-1][:-4])
            self._frames.set_limits(self.min_frame, self.max_frame)

            self.update()
    
    def _on_change_frame(self, new_value):
        if new_value < self.min_frame or new_value > self.max_frame:
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
        ###################
        # origin
        ###################
        # label = np.fromfile(join(ROOT, self.seq, 'labels', self.labels[frame]), dtype=np.int32)
        # #label = np.fromfile(join(ROOT, self.seq, '100', self.labels[frame]), dtype=np.int32)
        # label = label & 0xFFFF
    
        # ###################
        # # percent labels
        # ###################
        label_file = os.path.join(ROOT, str(self.seq), "0", str(frame).zfill(6)+'.npy')
        label = np.load(label_file, mmap_mode='r').astype(np.uint32)
        return label

    def ransac_dbscan(self, pcd, labels, idx):
        if np.array(pcd.points).shape[0] < self.ransac_points:
            return labels
        
        _, inliers  = pcd.segment_plane(distance_threshold=self.ransac_distance, ransac_n=self.ransac_points, num_iterations=self.ransac_iters, probability=0.99999999)
        if idx in inliers:
            labels[inliers] = labels[idx]
        else:
            mask = np.ones((labels.shape[0],))
            mask[inliers] = 0
            mask = mask>0

            db_pcd = o3d.geometry.PointCloud()
            db_pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)[mask])     
            db_labels = np.array(db_pcd.cluster_dbscan(eps=self.dbscan_eps, min_points=self.dbscan_min_points, print_progress=False))
            tmp_labels = np.zeros_like(labels)
            tmp_labels[mask] = db_labels
            mask1 = (tmp_labels==tmp_labels[idx])
            labels[mask1] = labels[idx]
        return labels

    def update(self):
        if self.dataset == 'SemanticPoss' or self.dataset == 'SemanticKITTI':
            print(join(ROOT, self.seq, 'velodyne',self.pcs[self.frame]))
            pc = np.fromfile(join(ROOT, self.seq, 'velodyne',self.pcs[self.frame]), dtype=np.float32)
            pc = pc.reshape((-1, 4))
            pc = pc[:,:3]
            pose0 = self.poses[self.frame - self.min_frame]


            label = self.get_label(self.frame)


            for i in range(1, self.scans):
                cur_frame = self.frame - i
                if cur_frame < self.min_frame:
                    break
                pose1 = self.poses[cur_frame - self.min_frame]
                pc1 = np.fromfile(join(ROOT, self.seq, 'velodyne',self.pcs[cur_frame]), dtype=np.float32)
                pc1 = pc1.reshape((-1, 4))
                pc1 = pc1[:,:3]

                label1 = self.get_label(cur_frame)



                pc1 = self.fuse_multi_scan(pc1, pose0, pose1)

                if len(pc1) > 0:
                    pc = np.concatenate((pc, pc1), axis=0)

                    if self.baseline:
                        label1 = np.zeros_like(label1)
                    
                    label = np.concatenate((label, label1), axis=0)

            # idx = (pc.shape[0]-1) - np.arange(pc.shape[0])
            # print(idx)
            # pc = pc[idx]
            # label = label[idx]

            truth = np.fromfile(join(ROOT, self.seq, 'labels', self.labels[self.frame]), dtype=np.int32)
            truth = truth & 0xFFFF

            # mask = label>0
            # pc, label = pc[mask], label[mask]


            if self.downsamping and pc.shape[0] > downsamping:
                idx = np.random.choice(pc.shape[0], downsamping, replace=False)
                print('idx:', idx.shape)
                pc = pc[idx]
                label = label[idx]

            
            ################
            # path dbscan
            ################
            region = np.array([2.5, 2.5])
            ratio_idxs = np.where(label>0)[0]
            for idx in ratio_idxs:
                dis = pc[:,:2] - pc[idx][:2]
 
                mask = np.all(((dis>-region) & (dis<region)), axis=1)
               
                idx1 = np.sum(mask[:idx])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[mask])
                label[mask] = self.ransac_dbscan(pcd, label[mask], idx1)
            ################
            #
            #################

            
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(pc)
            color = color_map[label]
            self.pcd.colors = o3d.utility.Vector3dVector(color)

            truth_mask = label>0
            print('Acc: {}'.format(np.sum(truth[truth_mask]==label[truth_mask])/np.sum(truth_mask)))
            print(np.sum(truth[truth_mask]==label[truth_mask]), np.sum(truth_mask))

            if BoundingBox is not None:
                pass #self.pcd = self.pcd.crop(BoundingBox)

                    
            
            print('pc shape: {}'.format(np.asarray(self.pcd.points).shape))

            self._scene.scene.clear_geometry()
            
            self._scene.scene.add_geometry('frame {}'.format(self.frame),self.pcd, self.geometry_render)

            self._filter_plane.checked = False
            self.ransac_pcd = self.pcd


    def ransac(self):
        _, inliers  = self.pcd.segment_plane(distance_threshold=self.ransac_distance, ransac_n=self.ransac_points, num_iterations=self.ransac_iters, probability=0.99999999)
        
        self.ransac_pcd = self.pcd.select_by_index(inliers, invert=True)
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry('frame {}'.format(self.frame),self.ransac_pcd, self.geometry_render)
        


    

    def fuse_multi_scan(self, points, pose0, pose):

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
    
    def parse_calibration(self, filename):
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

    def parse_poses(self, filename, calibration):
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

    def save_view(self):
        vis = self._scene
        fname='saved_view.pkl'
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
        fname='saved_view.pkl'
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
        