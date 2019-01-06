from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

from util import *


"""
function ：读取解析配置文件
:param cfgfile: cfg文件路径
:return: 返回blocks的list，每个blocks描述了将被构建的神经网络中的block。
        它的作用是解析cfg，并将每个块存储为字典。块的属性及其值在字典中作为键值对存储。在cfg解析过程中，
        将这些字典——在代码中称为block的变量，添加到名为blocks的列表变量中。函数将返回这个blocks
        （原文是block，但实际上它返回的是blocks列表）。
"""
def parse_cfg(cfgfile):

    file = open(cfgfile,'r')
    lines = file.read().split('\n')     # 将行存储在lines中作为一个列表
    lines = [x for x in lines if len(x) > 0]        # 去除空行
    lines = [x for x in lines if x[0] != '#']       # 去除注释
    lines = [x.rstrip().lstrip() for x in lines]    # 去除首位空白字符

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":      # 意味着新块的开始
            if len(block) != 0:     # 如果block不为空，意味着存储着之前块的值
                blocks.append(block)    #加入到list中
                block = {}          # 初始化block
            block["type"] = line[1:-1].rstrip()     # 将这个块的类型写入type
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()    # 去除空格写入字典中
    blocks.append(block)

    return blocks

def create_modules(blocks):
    """
    :param blocks: 使用parse_cfg返回的blocks列表作为输入
    :return:
    """
    net_info = blocks[0]        #获取输入的信息并且进行预处理，迭代blocks之前，定义变量存储网络信息
    module_list = nn.ModuleList()       # 相当于一个包含nn.Module对象的普通列表。
    prev_filters = 3        # kernel depth是前一层中过滤器的数量（或特征图的深度），使用这个变量持续跟踪
    output_filters = []     # route层的特征图来自前面的层（可能连接后的）的特征图。如果route层后有一个卷积层，
                            # 那么内核将应用在前面层的特征图上，那些特征图正是route层的特征图。因此，我们不仅
            # 需要跟踪前一层中的过滤器数量，还要跟踪前面所有层的。迭代时，我们将每个块的输出过滤器数添加到列表output_filters。
    filters = 0

    # 迭代块列表，并为每个块创建一个PyTorch模块
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()    # nn.Sequential类用于顺序执行一些nn.Module对象你查看一下cfg，你会意识到一个块可能包含多个层。 例如，除了卷积层以外，convolutional型块还具有批量标准化层以及Leaky ReLU激活层。 我们使用nn.Sequential和add_module函数将这些层串在一起。 例如，下面就是我们创建卷积和上采样层的代码。
                                    # 如果需要快速构建模型，则使用nn.Sequential()，如果需要自定义的比较多就使用nn.Module
        # 检查block的类型
        # 为这个block创建一个新的module
        # 添加到module_list

        if(x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2    # 如果使用padding，则利用卷积核大小算出pad的大小。
            else:
                pad = 0

            # 添加卷积层
            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias=bias)
            module.add_module("conv_{0}".format(index),conv)    # .format(index)函数可以将{}里面的数字替换

            # 添加Batch Norm 层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)        # 批归一化层，通过减少内部相关变量的移位来加速神经网络训练
                module.add_module("batch_norm_{0}".format(index), bn)

            # 检查激活函数，对于yolo来说Linear或者一个Leaky ReLU都是可以的
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        # 如果是一个上采样层，我们使用双线性上采样
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        # Route层/Shortcut层
        # 如果是一个route层
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            # route的起始点
            start = int(x["layers"][0])
            # 如果存在终止点的话
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:  # 位于Route层之后的卷积层将其内核应用于（可能连接的）前面层的特征图。以下代码更新filters变量以保存Route层输出的过滤器数量。
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        #shortcut corresponds to skip connection
        # shortcut层也使用空层，因为它执行非常简单的操作（相加）。没有必要更新filters变量，因为它仅仅将前一个层的特征图相加到后面的层的特征图上而已。
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo是detection层代码
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]    # 选择mask中的三个数字的anchors，输出三个值

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        # 循环结束时，储存一些记录
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    # 返回一个包含net_info和module_list的元组
    return (net_info, module_list)


"""
现在，空层可能看起来很奇怪，因为它什么都不做。Route层，就像任何其他层一样执行操作（使用前面的层/连接）。在PyTorch中，当我们定义一个新层时，它继承nn.Module，在nn.Module对象的forward函数写入层执行的操作。
为了设计Route块的层，我们必须构建一个nn.Module对象，它作为Layers的成员，使用Layers的属性值进行初始化。然后，我们可以在forward函数中编写代码来连接/获取特征图。最后，我们在网络的forward函数中执行该层的操作。
但是，如果连接代码相当简短（在特征图上调用torch.cat），那么设计一个如上所述的层将导致不必要的抽象，这只会增加代码。我们可以做一个空层来代替提出的Route层，然后直接在darknet的nn.Module对象的forward函数中执行连接。 
（如果你不明白这是什么意思，我建议你阅读PyTorch中如何使用nn.Module类，链接在本文末尾可以找到）https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#define-the-network
"""
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


"""我们定义了一个新的层DetectionLayer，它包含用于检测边界框的锚。
"""
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Darknet(nn.Module):

    # 定义网络
    def __init__(self,cfgfile):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info,self.module_list = create_modules(self.blocks)

    # 加载权重
    def load_weights(self,weightfile):

        if os.path.islink(weightfile):
            weightfile = os.readlink(weightfile)

        fp = open(weightfile,"rb")

        # 开头的5个值是header信息,文件的前160个字节存储5个int32值，构成文件头部
        # 1. Major version number
        # 2. Minor version number
        # 3. Subversion number
        # 4,5. 被网络所看到的图片(在训练时)
        header = np.fromfile(fp,dtype = np.int32,count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # 剩下的字节按照上文中提到的顺序表示权重，权重存储为float32或32位浮点数。把权重加载到一个np.ndarray中
        weights = np.fromfile(fp,dtype=np.float32)

        # 遍历权重文件，并将权重加载到我们的网络的模块中
        ptr = 0     # 使用ptr变量来跟踪我们在权重数组中的位置，现在，如果batch_normalize为True，按照如下方式去加载权重。
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # 如果module_type是convolutional，那么加载weights，否则忽略
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                # 根据是否使用批归一化导致的权重读取方式不一样。
                if(batch_normalize):
                    bn = model[1]

                    # 获取 Batch Norm 层的权重的数量
                    num_bn_biases = bn.bias.numel()

                    # 加载权重
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # 将加载的权重转换为模型的权重的维度
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # 将数据拷贝至模型中
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # biases的数量
                    num_biases = conv.bias.numel()

                    # 加载权重
                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr = ptr + num_biases

                    # 根据模型权重的维度来reshape加载的权重
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # 最后拷贝数据
                    conv.bias.data.copy_(conv_biases)

                # 为卷积层加载权重
                num_weights = conv.weight.numel()

                # 为权重做相同的操作
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)




    # 实现网络的前向传播
    # 需要三个参数，self，输入x，和CUDA->如果设置为True，则会使用GPU来加速正向传递
    def forward(self, x, CUDA):
        modules = self.blocks[1:]       # 这里使用的是self.blocks[1:]因为第0个元素时net块，不是前向传播的一部分
        outputs = {}    #为route层缓存输出，由于route和shortcut层需要前面的层的输出图，因此我们将每个层的输出特征图缓存在字典outputs中，键是层的索引，值是特征图

        write = 0       # 写入标志用于指示我们是否第一次检测。如果write为0，则表示收集器尚未初始化。如果是1，则意味着收集器已经初始化，我们可以将我们的检测图级联到它。
        for i,module in enumerate(modules):
            module_type = (module["type"])      # 模块已经按照配置文件中按顺序添加，可以简单的让输入通过每个模块获得输出

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]        # 当只有一个值时，如-4，输出当前层前面4层的值。

                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i       # 当有两个值时，如-1，16，则输出前一层和后16层的输出。

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1,map2),1)        # 需要连接两个特征图的 情况，使用cat函数并第二个参数设置为1，因为想要沿着深度连接特征图。
                                                        # 在Pytorch中，卷积层的输入和输出格式为B*C*H*W，深度对应于通道维度。

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]     # 跳过连接，from参数为-3，代表shortcut层的输出是通过将shortcut层的前一层和前面的第三层的特征图相加得到的。

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                # 获取输入维度
                inp_dim = int(self.net_info["height"])

                # 获取类别的数量
                num_classes = int(module["classes"])

                # 转换
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections,x),1)

            outputs[i] = x      # 输入如(1,10647,85)，1是图片数量，10647位边界框数量，85位边界框属性的数量

        return detections

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img,(416,416))     # 重设输入维度
    img_ = img[:,:,::-1].transpose((2,0,1))    # BGR -> RGB | HXWC -> CXHXW     当步长s>0时，list[i:j:1]相当于list[i:j]。当步长s<0时list[::-1]相当于list[-1:-len(list)-1:-1]
    img_ = img_[np.newaxis,:,:,:] / 255.0       # 添加一个channel在0处 | Normalise
    img_ = torch.from_numpy(img_).float()       # 转化为浮点类型
    img_ = Variable(img_)               # 转化为Variable类型
    return img_

model = Darknet("config/yolov3.cfg")
model.load_weights("weights/yolov3.weights")
inp = get_test_input()
pred = model(inp,torch.cuda.is_available())
# 输出的是一个(1,10647,85)维度的张量，对于一张图像，都有一个(10647,85)的表格，表格每一行代表一个边界框(4个bbox属性，1个目标分数和80个类别分数)
print(pred)
print(pred.shape)


# 测试网络模型代码
# blocks = parse_cfg("config/yolov3.cfg")
# print(create_modules(blocks))