#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021 by Luis J. Manso
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import cv2
import platform
import sys
import copy
import numpy as np
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.parse_objects import ParseObjects
from genericworker import *
import time
import platform

draw = True
# model = 'resnet'
# model = 'densenet'
# model = 'densenet_tuned'
model = 'densenet_tuned2'

REALSENSE = True
ZED = False
BORDER_THRESHOLD = 0.05

# GENERATE_VIDEO_MODE = True
GENERATE_VIDEO_MODE = False
if GENERATE_VIDEO_MODE is True:
    image_list = []
    draw = True

if REALSENSE:
    sys.path.append('/usr/local/lib/python3.6/pyrealsense2')
    import pyrealsense2.pyrealsense2 as rs
if ZED:
    import pyzed.sl as sl

assert REALSENSE ^ ZED, "Only one MUST be selected, either ZED or REALSENSE."

sys.path.append('../')
from parameters import parameters
from merge_skeletons import merge_skeletons


if REALSENSE:
    COLOUR_WIDTH = parameters.widths[parameters.camera_names.index(platform.node())]
    COLOUR_HEIGHT = parameters.heights[parameters.camera_names.index(platform.node())]
elif ZED:
    COLOUR_WIDTH = parameters.widths[parameters.camera_names.index(platform.node() + '_l')]
    COLOUR_HEIGHT = parameters.heights[parameters.camera_names.index(platform.node() + '_l')]

def cam_matrix_from_intrinsics(i):
    return np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]])


sys.path.append('/usr/local/lib/python3.6/pyrealsense2')

device = torch.device('cuda')

hostname = platform.node()

erode_kernel = np.ones((3, 3), np.uint8)
dilate_kernel = np.ones((40, 40), np.uint8)


def get_keypoint(humans, hnum, peaks):
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]  # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
        else:
            peak = (j, None, None)
            kpoint.append(peak)
    return kpoint


with open('../human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

print(f'------ model = {model} --------')
if model == 'resnet':
    MODEL_WEIGHTS = '../resnet18_baseline_att_224x224_A_epoch_249.pth'
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 224.
    HEIGHT = 224.
elif model == 'densenet':
    print('------ model = densenet--------')
    MODEL_WEIGHTS = '../densenet121_baseline_att_256x256_B_epoch_160.pth'
    OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 256.
    HEIGHT = 256.
elif model == 'densenet_tuned':
    MODEL_WEIGHTS = 'densenet121_baseline_att_416x416.pth'
    OPTIMIZED_MODEL = 'densenet121_baseline_att_416x416_trt.pth'
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 416.
    HEIGHT = 416.
elif model == 'densenet_tuned2':
    MODEL_WEIGHTS = 'densenet121_baseline_att_608x608.pth'
    OPTIMIZED_MODEL = 'densenet121_baseline_att_608x608_trt.pth'
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 608.
    HEIGHT = 608.

data = torch.zeros((1, 3, int(HEIGHT), int(WIDTH))).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1 << 25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()


def preprocess(image):
    global device
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


parse_objects = ParseObjects(topology, cmap_threshold=0.65, link_threshold=0.65)
# parse_objects = ParseObjects(topology, cmap_threshold=0.1, link_threshold=0.1)

if REALSENSE:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, int(COLOUR_WIDTH), int(COLOUR_HEIGHT), rs.format.bgr8, 30)
    pipeline.start(config)
    profile = pipeline.get_active_profile()
if ZED:
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print('Couldn\'t open ZED camera')
        exit(1)
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    fx, fy, cx, cy = calibration_params.left_cam.fx, calibration_params.left_cam.fy, calibration_params.left_cam.cx, calibration_params.left_cam.cy
    print(f'fx:{fx}  fy:{fy}  cx:{cx}  cy:{cy}')
    left_image = sl.Mat()
    right_image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()


def execute(img, src, t):
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    return counts, objects, peaks


def draw_arrow(image, i_w, i_h, ratio, xa, ya, xC, yC, xb, yb):
    tlx = int(ratio * xa) + int(i_w / 2)
    tly = int(i_h) - int(ratio * ya)
    brx = int(ratio * xb) + int(i_w / 2)
    bry = int(i_h) - int(ratio * yb)
    mx = int(ratio * xC) + int(i_w / 2)
    my = int(i_h) - int(ratio * yC)
    cv2.line(image, (tlx, tly), (mx, my), (255, 0, 0), 3)
    cv2.line(image, (mx, my), (brx, bry), (0, 0, 255), 3)


last_sent = time.time()


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map):
        super(SpecificWorker, self).__init__(proxy_map)
        self.host = platform.node()

        self.backSub = cv2.createBackgroundSubtractorKNN()

        while True:
            self.messages = self.compute()
            time.sleep(0.005)
            for host, message in self.messages.items():
                try:
                    self.pose_proxy.sendData(host, message, f'live_{host}')
                except Ice.ConnectionRefusedException as e:
                    print('Cannot connect to host, waiting a few seconds...')
                    print(e)
                    time.sleep(3)
                except Ice.ConnectionLostException as e:
                    print('Cannot connect to host, waiting a few seconds...')
                    print(e)
                    time.sleep(3)
                except Ice.ConnectionTimeoutException as e:
                    print('timeout')
                    print(e)
                    time.sleep(3)
                except Ice.UnknownException as e:
                    print('Cannot connect to host, waiting a few seconds...')
                    print(e)
                    time.sleep(3)

    def __del__(self):
        print('SpecificWorker destructor')

    def setParams(self, params):
        return True

    def compute(self):
        if REALSENSE:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            return {hostname: self.deep_compute(color_image, hostname)}
        elif ZED:
            if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
                print('Can\'t grab.')
                return {}
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            left = left_image.get_data()
            right = right_image.get_data()
            return {hostname + '_l': self.deep_compute(left, hostname + '_l'),
                    hostname + '_r': self.deep_compute(right, hostname + '_r')}
        elif CV2:
        	frame, ret = grabber.grab()
               return {hostname: self.deep_compute(color_image, hostname)}
 
    def deep_compute(self, color_image, effective_hostname):
        global last_sent
        data_to_publish = []
        t = time.time()
        r_s = parameters.r_s[parameters.camera_names.index(effective_hostname)]
        r_w = parameters.r_w[parameters.camera_names.index(effective_hostname)]
        c_s = parameters.c_s[parameters.camera_names.index(effective_hostname)]
        c_w = parameters.c_w[parameters.camera_names.index(effective_hostname)]
        cropped = color_image[int(r_s):int(r_s + r_w), int(c_s):int(c_s + c_w)]

        # fgMask = self.backSub.apply(cropped)
        # fgMask = cv2.erode(fgMask, erode_kernel)
        # fgMask = cv2.dilate(fgMask, dilate_kernel)
        # cropped[fgMask<10] = 0

        img = cv2.resize(cropped, dsize=(int(WIDTH), int(HEIGHT)), interpolation=cv2.INTER_AREA)
        counts, objects, peaks = execute(img, color_image, t)

        if draw or GENERATE_VIDEO_MODE:
            ret = copy.deepcopy(color_image)
            color = (0, 255, 0)
            for i in range(counts[0]):
                keypoints = get_keypoint(objects, i, peaks)
                for j in range(len(keypoints)):
                    if keypoints[j][1]:
                        x = keypoints[j][2] * c_w + c_s
                        y = keypoints[j][1] * r_w + r_s
                        cv2.circle(ret, (int(x), int(y)), 1, color, 2)
                        cv2.putText(ret, "%d" % int(keypoints[j][0]), (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 255), 1)

            if draw:
                cv2.imshow("cropped" + ' ' + effective_hostname, cropped)

        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)
            kps = dict()
            for kp in range(len(keypoints)):
                centre = keypoints[kp]
                if centre[1] and centre[2]:
                    if centre[2] > BORDER_THRESHOLD and centre[1] > BORDER_THRESHOLD:
                        if centre[2] < 1. - BORDER_THRESHOLD and centre[1] < 1. - BORDER_THRESHOLD:
                            cx = centre[2] * c_w + c_s
                            cy = centre[1] * r_w + r_s
                            kps[kp] = [kp, float(cx), float(cy), 1, 1]
            data_to_publish.append(kps)

        data_to_publish = merge_skeletons(data_to_publish)

        if parameters.tracker_sends_only_one_skeleton:
            max_joints = 0
            max_joints_skeleton = None
            for data in data_to_publish:
                if len(data) > max_joints:
                    max_joints_skeleton = data
            if max_joints_skeleton:
                data_to_publish = [max_joints_skeleton]

        if draw:
            cv2.imshow("test" + ' ' + effective_hostname, ret)
            if cv2.waitKey(1) % 256 == 27:
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                sys.exit(0)
        elif GENERATE_VIDEO_MODE:
            image_list.append(ret)
            if len(image_list) == 1000:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 20, (ret.shape[1], ret.shape[0]))
                for im in image_list:
                    out.write(im)
                out.release()
                sys.exit(10)

        return json.dumps(data_to_publish)
