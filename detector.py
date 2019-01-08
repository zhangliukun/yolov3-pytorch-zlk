from __future__ import division

import datetime
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
import sys


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="data/test_folder/", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",default="config/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",default="weights/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. "
                                            "Decrease to increase speed",default="416", type=str)    # 输入图像的分辨率

    return parser.parse_args()

args = arg_parse()
images = args.images

batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80    #For COCO
classes = load_classes("data/coco.names")

# 构建神经网络
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

args.det = "det"

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

#Set the model in evaluation mode ，可能对于一些Dropout有影响
model.eval()

read_dir = time.time()  # 测量时间检查点

# 检测阶段
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()

# 如果由det标志指定的检测目录不存在，创建。
if not os.path.exists(args.det):
    os.makedirs(args.det)

# 使用opencv加载图像
load_batch = time.time()    # 时间检查点

# opencv将图像加载为numpy数组，颜色通道为BGR。Pytorch图像输入格式为(批量,通道,高，宽)，通道顺序为RGB，因此需要编写prep_image函数将numpy数组转为Pytorch格式
loaded_ims = [cv2.imread(x) for x in imlist]

# 除了转换后的图像，我们还维护了一张原始图像列表，以及包含原始图像尺寸的列表im_dim_list。
# PyTorch Variables for images
# Make an iterator that computes the function using arguments from each of the iterables.  Stops when the shortest iterable is exhausted.
# map函数，map(func, *iterables) --> map object，第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

# 列出原始的图片包含的维度
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)    # [602,452] -> [602,452,602,452]

if CUDA:
    im_dim_list = im_dim_list.cuda()

# 创建批
leftover = 0
if (len(im_dim_list) % batch_size):
   leftover = 1

if batch_size != 1:
   num_batches = len(imlist) // batch_size + leftover
   im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                       len(im_batches))]))  for i in range(num_batches)]

# 检测循环
# 我们按批迭代，生成预测结果，并把执行检测的所有图像的预测结果的张量（它的形状是D x 8，，来自write_results函数的输出）连接起来。
# 对于每个批，我们将测量检测所花费的时间，即获取输入和生成write_results函数输出之间的时间。在由write_prediction返回的输出中，
# 其中一个属性是批中图像的索引。我们对该特定属性（索引）进行转换，使其成为imlist（该列表包含所有图像的地址）中图像的索引。
# 之后，我们会打印每个检测的时间以及每个图像中检测到的目标。
# 如果批的write_results函数的输出是int（0），意味着没有检测，我们使用continue继续跳过剩下的循环。
write = 0
start_det_loop = time.time()

#prev_time = time.time()

for i, batch in enumerate(im_batches):
    #load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    with torch.no_grad():
        label = Variable(batch)

    prediction = model(label, CUDA)      #一张图片的输出结果[1,10647,85]，其中10647代表bbox数量

    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)       #输出维度(3,8)

    end = time.time()

    # current_time = time.time()
    # inference_time = datetime.timedelta(seconds=current_time - prev_time)
    # prev_time = current_time
    # print('\t+ Batch %d, Inference Time: %s' % (i, inference_time))

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 将batch中的索引转化为imlist中的索引

    if not write:                      #If we have't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()       # 确保CUDA内核与CPU同步。否则，CUDA内核会在GPU作业排队后立即将控制返回给CPU，
                # 这时GPU作业尚未完成（异步调用）。如果在GPU作业实际结束之前end = time.time（）被打印出来，这可能会导致错误的时间。

    # 现在output张量拥有了所有图像的输出。开始在图像上面绘制边界框
    try:
        output
    except NameError:
        print("No detections were made")    # 检测有没有检测结果
        exit()

    # 我们输出张量中包含的预测是相对于网络的输入图像的尺寸的数据，而不是图像的原始大小。因此，在我们绘制边界框之前，
    # 让我们将每个边界框的角点的属性转换为图像的原始尺寸。仅仅将它们重新缩放到输入图像的尺寸并不适用。我们首先需要转换边界框的坐标，
    # 使得它的测量是相对于填充图像中的原始图像区域。
im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

# 现在，我们的坐标的测量是在填充图像中的原始图像区域上的尺寸。但是，在函数letterbox_image中，我们通过缩放因子调整了图像的两个维度
# （记住，这两个维度的调整都用了同一个因子，以保持宽高比）。我们现在撤销缩放以获得原始图像上边界框的坐标。
output[:, 1:5] /= scaling_factor

# 让我们现在对那些框边界在图像边界外的边界框进行裁剪。
for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

# 如果图像中的边界框太多，将它们全部绘制成同一种颜色可能不大好。将此文件下载到您的检测器文件夹。这是一个pickle文件，它包含许多可随机选择的颜色。
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("config/pallete", "rb"))

# 现在让我们编写一个用于绘制边界框的函数。
draw = time.time()

# 上面的函数使用从colors中随机选择的颜色绘制一个矩形框。它还在边界框的左上角创建一个填充的矩形，并将检测到的目标的类写入填充矩形中。使用cv2.rectangle函数的-1参数来创建填充的矩形。
# 我们在局部定义write函数，以便它可以访问colors列表。我们也可以将colors作为参数，但是这会让我们每个图像只能使用一种颜色，这会破坏我们想要使用多种颜色的目的。
def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]        # 图片的索引
    cls = int(x[-1])                # 图片的类别
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img

# 代码修改了loaded_ims内的图像。
list(map(lambda x: write(x, loaded_ims), output))

# 通过在图像名称前添加“det_”前缀来保存每张图像。我们创建一个地址列表，并把包含检测结果的图像保存到这些地址中。
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

# 最后，将带有检测结果的图像写入det_names中的地址。
list(map(cv2.imwrite, det_names, loaded_ims))
end = time.time()

"""
打印时间总结
在我们的检测器结束时，我们将打印一份总结，其中包含哪部分代码需要多长时间才能执行。当我们需要比较不同的超参数如何影响检测器的速度时，
这非常有用。可以在命令行上执行脚本detection.py时设置超参数，如批的大小，目标置信度和NMS阈值（分别通过bs，confidence，nms_thresh标志传递）。
"""
print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(imlist)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()