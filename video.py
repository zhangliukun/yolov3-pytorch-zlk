from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",default="config/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",default="weights/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. "
                                            "Decrease to increase speed",default="416", type=str)    # 输入图像的分辨率
    parser.add_argument("--video", dest="videofile", help="Video file to     run detection on", default="video.avi",
                        type=str)

    return parser.parse_args()


args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("data/coco.names")

# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()


def write(x, results):
    # 只显示行人
    cls = int(x[-1])
    if cls != 0:
        return results
        #color = random.choice(colors[1:])
    else:
        color = colors[0]

    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results

    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img


# Detection phase

videofile = args.videofile  # or path to the video file.

cap = cv2.VideoCapture(videofile)

#cap = cv2.VideoCapture(0)  #for webcam

assert cap.isOpened(), 'Cannot capture source'

args.det = "det/video/"
if not os.path.exists(args.det):
    os.makedirs(args.det)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
print(args.det+'%s.avi' % videofile.split('/')[-1].split('.')[0])
out = cv2.VideoWriter(args.det+'%s.avi' % videofile.split('/')[-1].split('.')[0], fourcc, 15.0, (960, 540))


frames = 0
start = time.time()
pre_time = start

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        img = prep_image(frame, inp_dim)
        #        cv2.imshow("a", frame)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img, volatile=True), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
            #cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        classes = load_classes('data/coco.names')
        colors = pkl.load(open("config/pallete", "rb"))

        # 将框框画入原图上面
        currentFps = 1/(time.time()-pre_time)
        list(map(lambda x: write(x, frame), output))

        # 照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
        cv2.putText(frame, "{:5.2f}".format(currentFps), (20, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 5)

        #cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1

        print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
        print("single image FPS is {:5.2f}".format(currentFps))

        #orig_im = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_CUBIC)
        out.write(frame)

        pre_time = time.time()

        if CUDA:
            torch.cuda.synchronize()  # 确保CUDA内核与CPU同步。否则，CUDA内核会在GPU作业排队后立即将控制返回给CPU，
            # 这时GPU作业尚未完成（异步调用）。如果在GPU作业实际结束之前end = time.time（）被打印出来，这可能会导致错误的时间。

    else:
        break