import os
import argparse
import json
import select
from os.path import join

import numpy as np
from scipy.spatial.transform import Rotation as R

import pickle as pkl
from tqdm import tqdm 

import sys
import termios
import tty

import cv2

parser = argparse.ArgumentParser(description="Video preprocessor")

parser.add_argument('--input', required=True, help='Путь к кадрам видео')
parser.add_argument('--output', required=True, help='Путь для сохранения pkl')

parser.add_argument("-s", "--sequence", type=str, default="0", 
                    help="Sequence number (Default 0)")
parser.add_argument("-f", "--start_frame", type=str, default="0",
                    help="Frame to start at (Default 0)")
parser.add_argument("-c", "--color_type", type=str, default="classId", 
                    help="Color map to use for coloring boxes Options: [isOccluded, classId] (Default classId)")
parser.add_argument("-l", "--log", type=str, default="",
                    help="Logs point cloud and bbox annotations to file for external usage")
parser.add_argument("-n", "--namespace", type=str, default="coda",
                    help="Select a namespace to use for published topics")

def get_key():
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return ch

def vis_annos_rviz(args):
    namespace = args.namespace

    IMAGE_PATH = args.input
    OUTPUT_DIR = args.output

    if not os.path.exists(IMAGE_PATH):
        print(f"Папка {IMAGE_PATH} не существует!")
        exit()

    i = 0
    for filename in os.listdir(IMAGE_PATH):
        file_path = os.path.join(IMAGE_PATH, filename)
        
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            
            if image is not None:
                print(f"Успешно загружено изображение: {filename}")
            else:
                print(f"Не удалось загрузить изображение: {filename}")
        
        # we want to save:
        # cam0, stereo, bbox_3d_json, pose, and lidar_ts

        out_dict = {}
        out_dict['cam0'] = image
        out_dict['bbox_3d'] = 'none'
        out_dict['position'] = np.array([10,10,10])
        out_dict['rotation'] = np.array(3.14)
        out_dict['timestamp'] = i


        # Make trajectory here
        if not os.path.exists(OUTPUT_DIR):
            print("Output image dir for %s does not exist, creating..."%OUTPUT_DIR)
            os.makedirs(OUTPUT_DIR)

        with open(join(OUTPUT_DIR, f'{i}.pkl'), 'wb') as handle:
            pkl.dump(out_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
        i += 1


if __name__ == '__main__':
    args = parser.parse_args()

    vis_annos_rviz(args)
