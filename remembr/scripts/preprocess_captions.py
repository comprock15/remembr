import argparse
import re
from io import BytesIO
import os, os.path as osp

import requests
from PIL import Image
import numpy as np
import sys

# load this directory
sys.path.append(sys.path[0] + '/..')
from captioners.vila_captioner import VILACaptioner
from utils.util import get_frames
import pickle as pkl
from PIL import Image as PILImage

from langchain_huggingface import HuggingFaceEmbeddings
import glob
from scipy.spatial.transform import Rotation
import shutil
import json

import tqdm

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def run_video_in_segs(args):

    # SEQUENCE_ID=args.seq_id
    PKL_DATA_PATH = args.data_path
    print(PKL_DATA_PATH)
    # load folders
    pkl_files = glob.glob(os.path.join(PKL_DATA_PATH, '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    times = [float(x.split('/')[-1][:-4]) for x in pkl_files]

    segments = []
    current_segment = []
    time_start = times[0]
    for t, file in zip(times, pkl_files):
        if t - time_start >= args.seconds_per_caption:
            # Then start over. Add the previous group. This item is the first of the new group
            segments.append(current_segment)
            current_segment = [file]
            time_start = t
        else:
            # Add current file to group
            current_segment.append(file)

    embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
    vila_model = VILACaptioner(args)

    # if exists, then exit
    # captions_location = f'./data/{SEQUENCE_ID}/captions'
    captions_location = args.out_path
    # if os.path.exists(captions_location):
    #     # exit()
        # shutil.rmtree(captions_location, ignore_errors=True) #удаляет все в директории 
    print(captions_location)
    os.makedirs(captions_location, exist_ok=True)

    outputs = []

    for i, file_names in tqdm.tqdm(enumerate(segments), total=len(segments)):
        images = []
        # depth = []
        # bboxes = []
        position = []
        rotation = []
        timestamp = []
        cam0 = []


        for file in file_names:
            with open(file, 'rb') as f:
                data = pkl.load(f)
                data['cam0'] = data['cam0'][:, :, ::-1]

                images.append(PILImage.fromarray(data['cam0'].astype('uint8'), 'RGB'))
                # depth.append(data['stereo'])
                # bboxes.append(data['bbox_3d'])
                position.append(data['position'])
                rotation.append(data['rotation'])
                timestamp.append(data['timestamp'])
                cam0.append(data['cam0'])

        
        position = np.array(position)
        rotation = np.array(rotation)
        # rotation = Rotation.from_quat(rotation).as_euler('xyz', degrees=True)
        timestamp = np.array(timestamp)

        # let's sample the images down to args.num_video_frames
        # images = images[::30//args.num_video_frames]
        # Берем фиксированное количество кадров, равномерно распределенных
        if len(images) > args.num_video_frames:
            step = len(images) // args.num_video_frames
            images = images[::step][:args.num_video_frames]
        else:
            # Если кадров меньше нужного, оставляем как есть
            images = images

        out_text = vila_model.caption(images)

        print(out_text)
        filename_start = os.path.basename(file_names[0])
        filename_end = os.path.basename(file_names[-1])


        text_embedding = embedder.embed_query(out_text)

        
        entity = {
            'id': file_names[0],
            'position': position.mean(axis=0),
            'theta': 3.14, # TEMPORARY: We are not using rotation information yet, so just leaving a placeholder
            'time': timestamp.mean(),
            'caption': out_text,
            'file_start': filename_start,
            'file_end': filename_end,
            'text_embedding': text_embedding,
            'cam0': cam0[0],
        }

        outputs.append(entity)


    # now save the outputs into a json
    with open(os.path.join(captions_location, f'captions_{args.captioner_name}_{args.seconds_per_caption}_secs.json'), 'w') as f:
        json.dump(outputs, f, cls=NumpyEncoder)


if __name__ == "__main__":

    default_query = "<video>\n You will be shown videos where people are busy with something. \
        Please describe in detail what you see during the few seconds of the video. \
        Specifically focus on the people, objects, environmental features, events/ectivities, and other interesting details. Think step by step about these details and be very specific."

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-3b")
    # parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-13b")
    # parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--seq_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="./coda_data")
    parser.add_argument("--out_path", type=str, default="./data/captions")
    parser.add_argument("--captioner_name", type=str, default="VILA1.5-3b")

    parser.add_argument("--seconds_per_caption", type=int, default=3)

    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str, default=default_query)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()


    # add some rules here
    if 'Efficient-Large-Model/VILA1.5-40b' in args.model_path:
        args.conv_mode = 'hermes-2'
    elif 'Efficient-Large-Model/VILA1.5' in args.model_path:
        args.conv_mode = 'vicuna_v1'
    elif 'Llama' in args.model_path:
        args.conv_mode = 'llama_3'
    else:
        # trust the default conv_mode
        # args.conv_mode = args.conv_mode
        args.conv_mode = 'vicuna_v1'

    run_video_in_segs(args)


# python -W ignore caption_segments.py --video-file "/home/aanwar/projects/memory_nav/foundation-nav/tools/isaac/data/102344094/path_vis/output.avi"     --query "<video>\n Please describe what you see in the few seconds of the video." 
