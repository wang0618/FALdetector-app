import io
import sys
import argparse
from os import path

here_dir = path.dirname(path.abspath(__file__))
FAL_dir = path.join(here_dir, 'FALdetector')
sys.path.insert(0, FAL_dir)

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from global_classifier import classify_fake, load_classifier
from networks.drn_seg import DRNSeg
from utils.tools import *
from utils.visualize import *

import pywebio
from pywebio import start_server
from pywebio.input import *
from pywebio.output import *

readme_md = open(path.join(here_dir, 'README.md')).read()

global_model = None
local_model = None
device = 'cpu'


def load_local_model(model_path):
    global local_model

    model = DRNSeg(2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()

    return model


def local_detector(model, img_file, no_crop=False):
    # Data preprocessing
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    im_w, im_h = Image.open(img_file).size
    if no_crop:
        face = Image.open(img_file).convert('RGB')
    else:
        faces = face_detection(img_file, verbose=False)
        if len(faces) == 0:
            print("no face detected by dlib, exiting")
            sys.exit()
        face, box = faces[0]
    face = resize_shorter_side(face, 400)[0]
    face_tens = tf(face).to(device)

    # Warping field prediction
    with torch.no_grad():
        flow = model(face_tens.unsqueeze(0))[0].cpu().numpy()
        flow = np.transpose(flow, (1, 2, 0))
        h, w, _ = flow.shape

    # Undoing the warps
    modified = face.resize((w, h), Image.BICUBIC)
    modified_np = np.asarray(modified)
    reverse_np = warp(modified_np, flow)
    reverse = Image.fromarray(reverse_np)

    flow_magn = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    cv_out = get_heatmap_cv(modified_np, flow_magn, max_flow_mag=7)
    heatmap = Image.fromarray(cv_out)

    return modified, reverse, heatmap


def main():
    set_auto_scroll_bottom(False)

    put_markdown(readme_md)

    put_markdown('## Have a try')

    while True:
        img = file_upload('Select image to upload', accept="image/jpeg",
                          help_text="目前只接受 jpg 文件格式；如果文件过大，上传可能较慢", required=True)

        put_text("上传成功, 处理中 ...")

        img_file = io.BytesIO(img['content'])

        try:
            prob = classify_fake(global_model, img_file, no_crop=False)
            put_markdown("Probibility being modified by Photoshop FAL: **{:.2f}%**".format(prob * 100))

            images = local_detector(local_model, img_file, no_crop=False)
        except SystemExit:
            put_text("错误🙅：没有检测到人脸")
            put_markdown('----')
            continue

        files = [io.BytesIO() for _ in images]
        for idx, im in enumerate(images):
            im.save(files[idx], format='jpeg')

        put_markdown('处理结果，从左往右依次是： 上传的图片|修改的部分|还原的图片')
        put_image(files[0].getvalue(), title='origin image')

        put_image(files[2].getvalue(), title='heatmap')

        put_image(files[1].getvalue(), title='undo result')

        put_markdown('----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=80, type=int, help="listen port")
    args = parser.parse_args()

    pywebio.enable_debug()
    global_model = load_classifier(path.join(FAL_dir, 'weights', 'global.pth'), 0)
    local_model = load_local_model(path.join(FAL_dir, 'weights', 'local.pth'))

    start_server(main, port=args.port, debug=True, auto_open_webbrowser=False)
