#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from os.path import dirname, isdir, join

"""
This is the config file for the project.
You can change the path here.
"""

root = dirname(__file__)


class GPU23LU:
    @property
    def fps5000_video_folder(self):
        return join(root, "dataset/1-Videos-EvNurall/")

    @property
    def fastec_video_folder(self):
        return join(root, "dataset/2-Fastec-Simulated/")

    @property
    def timelens_pp_root(self):
        return join(root, "dataset/3-TimeLens++/")

    @property
    def timelens_root(self):
        return join(root, "dataset/9-TimeLens/hsergb/")

    @property
    def adobe240fps_root(self):
        return join(root, "dataset/2-240fps-Videos/1-Adobe240Fps/OriginalVideo/frame/")

    @property
    def gopro_root(self):
        return join(root, "dataset/2-240fps-Videos/2-GoPro/")

    @property
    def color_event_dataset(self):
        return join(root, "dataset/8-CED-dataset/")


global_path = GPU23LU()
