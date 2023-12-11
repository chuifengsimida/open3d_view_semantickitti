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

import pyautogui

npz_file = 'mid_features/2023_03_23_15_04_16_con2_7layers.npz'

data = np.load(npz_file)

names = ['encode_2', 'encode_4', 'encode_8', 'decode_16', 'decode_8', 'decode_4', 'decode_2', 'pred', 'label']


root = npz_file[:-4]
if not os.path.exists(root):
    os.mkdir(root)






class ShowWindow:
    def __init__(self) -> None:
        
        self.window = gui.Application.instance.create_window(
                    "view semantic segmentation", 1027, 768
                )


        self._scene = gui.SceneWidget()
        self._scene_rendering = rendering.Open3DScene(self.window.renderer)
        self._scene_rendering.set_background(np.array([255,255,255,255])/255.0)
        self._scene.scene = self._scene_rendering
        self.geometry_render = o3d.visualization.rendering.MaterialRecord()
        self.geometry_render.point_size = 2


        em = self.window.theme.font_size
        self._settings_panel = gui.Vert(
                    0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self._next = gui.Button('Next')
        self._next.set_on_clicked(self._on_next)
        
        self._ScreenShot = gui.Button('Screen Shot')
        self._ScreenShot.set_on_clicked(self._screen_shot)
        
        self._settings_panel.add_child(self._next)
        self._settings_panel.add_child(self._ScreenShot)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)
        self.window.add_child(self._settings_panel)

        self.idx = -1
        
        self._on_next()
    
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

    def _on_next(self):
        self.idx += 1
        if self.idx == len(names):
            self.idx = 0
        key = names[self.idx]

        min_bound = np.array([50, 50, 5])
        mask = np.all(data['pc']>-min_bound, 1) & np.all(data['pc']<min_bound, 1)
        print(mask.shape)
        
        pcd = o3d.geometry.PointCloud()
        
        pcd.points = o3d.utility.Vector3dVector(data['pc'][mask])
        labels = np.mean(data[key], 1).reshape(-1, 1)#data['label']
        labels = labels[mask]
        

        print('Debug:', np.max(data[key], 1).reshape(-1, 1).shape, labels.shape)
        print(np.max(labels, 0),np.min(labels, 0))
        norm_labels = (labels - np.min(labels, 0))/(np.max(labels, 0)-np.min(labels, 0))
        colors = np.concatenate([norm_labels, 1-norm_labels, 0.5+norm_labels], axis=1)

        pcd.colors = o3d.utility.Vector3dVector(colors)    

    
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry(key,pcd, self.geometry_render)

        print(key+":", data[key].shape)
        
    def _screen_shot(self):
        key = names[self.idx]
        myScreenshot = pyautogui.screenshot()
        print('Saving ', os.path.join(npz_file[:-4], '{}.png'.format(key)))
        myScreenshot.save(os.path.join(npz_file[:-4], '{}.png'.format(key)))


 

if __name__=='__main__':
    gui.Application.instance.initialize()
    w = ShowWindow()
    gui.Application.instance.run()



