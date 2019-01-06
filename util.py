from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    """
    此函数将输入的特征检测图转换成二维张量，其中张量的每一行对应于边界框的属性，按以下顺序排列。
    前三行是在(0,0)处的三个B.Box，后三行是在(0,1)处的三个B.Box，依次类推
    :param prediction: 输出
    :param inp_dim: 输入图像尺寸
    :param anchors:
    :param num_classes:
    :param CUDA:
    :return:
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # 在Pytorch中的tensor的有些操作并不改变tensor的内容，但是只是改变了字节位置的索引，这些操作包括：narrow(),view(),expand()和transpose()
    # 例如，当你使用transpose()操作时，并不会使用new layout生成新的tensor。只是改变了Tensor中的元信息所以offset和stride得到新的形状，这个
    # transposed tensor和原始的tensor实际上是共享内存的。
    prediction = prediction.view(batch_size,bbox_attrs*num_anchors,grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()     # 有些通过transpose得到的tensor的memort layout is different than a tensor of same shape made from scratch,内存是一块的但是元素的次序不同。When you call contiguous(), it actually makes a copy of tensor so the order of elements would be same as if tensor of same shape created from scratch.
    prediction = prediction.view(batch_size,grid_size*grid_size*num_anchors,bbox_attrs)     # (1,507,85)代表一张图片，anchors总共的数量为507每个grid有三个anchors，每个bbox的属性：80个类别+1个置信度+4个坐标值。

    anchors = [(a[0]/stride,a[1]/stride) for a in anchors] # anchor的尺寸是根据net块的height和width属性。这些属性是输入图像的尺寸，
                                                    # 它比检测图大（输入图像是检测图的stride倍）。因此，比如通过检测特征图的stride划分anchor

    # 对center_X,center_Y坐标和 object confidence进行Sigmoid变换
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # 将网格偏移添加到预测的中心坐标
    grid = np.arange(grid_size)     # np.arange(3)  => array([0, 1, 2])
    a,b = np.meshgrid(grid,grid)    # 可以生成网格，对于计算网格上面的非常有效

    x_offset = torch.FloatTensor(a).view(-1,1)      # torch.Tensor是默认的FloatTensor的简称，将np转为tensor
    y_offset = torch.FloatTensor(b).view(-1,1)      # view的作用是将一个多行的Tensor拼接成一行。-1的意思就是让库自行计算行数或者列数。

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # Repeats this tensor along the specified dimensions. Unlike :meth:`~Tensor.expand`, this function copies the tensor's data.
    # 比如repeat(4,2)，则是行重复4次，列重复2次
    # unsqueeze(0) 在指定位置第0维插入一个维度
    x_y_offset = torch.cat((x_offset,y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    # 将anchor应用于边界框的尺寸
    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # 将Sigmoid激活应用于类别分数
    prediction[:,:,5:5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))

    # 将检测图像调整为输入图像的大小，此处的边界框属性根据特征图的大小而定的，如果输入图像是416*416，我们将这些属性乘以32或变量stride
    prediction[:,:,:4] *= stride

    return prediction

def write_results(prediction,confidence,num_classes,nms_conf = 0.4):
    """
    我们必须将输出结果根据目标分数阈值和非最大值抑制来获得true检测结果
    :param prediction:  输入预测
    :param confidence:  置信度（目标分数阈值）
    :param num_classes:     类别的数量
    :param nms_conf:    NMS IOU阈值
    :return:
    """

    # 目标置信度阈值，对于每个具有低于阈值的目标分数的边界框，我们将它的每个属性（边界框的整个行）的值设置为0
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # 执行非最大值抑制，我们现在具有的边界框属性由中心坐标以及边界框的高度和宽度描述。但是，使用每个框的一对角点的坐标
    # 来计算两个框的IoU更容易。因此，我们将框的（中心x，中心y，高度，宽度）属性转换为（左上角x，左上角y，右下角x，右下角y）
    box_corner = prediction.new(prediction.shape)   # Constructs a new tensor of the same data type as self tensor.
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    # 每幅图像中的true检测结果的数量可能不同。例如，批量大小为3，图像1,2和3分别具有5个，2个和4个true检测结果。因此，一次只能对一张图像
    # 进行置信度阈值和NMS。这意味着，我们不能向量化所涉及的操作，并且必须在prediction的第一维（包含批量中的图像索引）上进行循环。
    batch_size = prediction.size(0)

    write = False   # 如前所述，write标志用于指示我们尚未初始化output，我们将使用张量来保存整个批量的true检测结果。

    for ind in range(batch_size):
        image_pred = prediction[ind]  # image Tensor
        # confidence threshholding
        # 每个边界框行有85个属性，其中80个是类别分数。此时，我们只关心具有最大值的类别分数。因此，我们从每一行中删除80个类别的分数，并添加具有最大值的类别的索引，以及该类别的类别分数
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # 清除小于阈值的目标置信度的边界框
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue    # try-except用于处理没有检测到的情况，使用contine来跳过

        # For PyTorch 0.4 compatibility
        # Since the above code with not raise exception for no detection
        # as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue

        # 获取图片中检测到的不同的类
        img_classes = unique(image_pred_[:,-1])     # -1索引保存着类别的索引


        # 提取特定的类的检测结果，用cls表示
        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # 对detections进行排序这样有最大的物体置信度的实体排在最上面。
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # detections的数量

            # 执行NMS
            for i in range(idx):
                # 获取循环中我们正在查看的框之后的所有框的IOU
                try:
                    # bbox_iou，第一个参数是由循环中变量i索引的边界框行，第二个参数是包含多行边界框的张量
                    # 函数bbox_iou的输出是一个张量，它包含第一个参数所表示的边界框与第二个参数中的每个边界框的IoU。
                    # 下式给出了索引为i的框与所有索引都高于i的边界框的IoU。
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break
                    # 放try-catch中是因为循环被设计为运行idx迭代，我们会从image_pred_class中删除多个边界框。循环中删元素可能会错位。这时就跳出循环。
                except IndexError:
                    break

                # 每次迭代，任何具有索引大于i的的边界框，若其IoU大于阈值nms_thresh（具有由i索引的框），则该边界框将被去除。
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            # 写入预测结果，函数write_results输出形状为D x 8的张量。这里D是所有图像的true检测，每个检测由一行表示。每个检测有8个属性，
            # 即检测的图像在所属批次中的索引，4个角坐标，目标分数，最大置信度类别的分数以及该类别的索引。

            # 和以前一样，除非我们有一个检测分配给它，否则我们不会初始化输出张量。一旦它被初始化，我们把后续的检测与它连接。
            # 我们使用write标志来指示张量是否已经初始化。在遍历类的循环结束时，我们将检测结果添加到张量output中。
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)

            # 对于图片中类cls的检测重复batch_id
            #  Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind,image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    # 在函数结束时，我们检查输出是否已经被初始化。如果它没有意味着在该批次的任何图像都没有检测到任何目标。在这种情况下，我们返回0。
    try:
        return output       # 返回了一个预测张量，每个预测列为行。
    except:
        return 0




# 由于同一个类可以有多个true的检测结果，我们使用一个称为unique的函数来获取任何给定图像中存在的类
def unique(tensor):
    tensor_np = tensor.cpu().numpy()        # 如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1,box2):
    """
    :param box1:
    :param box2:
    :return: 返回两个边界框的IoU
    """
    # 获取bounding boxes的坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # 获得交集矩形的坐标
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    # 交集区域
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) \
                 * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,min=0) # clamp将所有的input限制到[min,max]之间。

    # 合并区域
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

# 它返回一个将每个类的索引映射到名称字符串的字典。
def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

# 调整图像的大小，保持宽高比一致，并用颜色（128,128,128）填充空白的区域。
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding
        使用padding来使得改变图像大小且比例不变
    '''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas

# 以OpenCV图像作为输入，并将它转换为网络输入的格式。
def prep_image(img, inp_dim):
    """
    为神经网络准备图片输入

    Returns a Variable
    """

    img = letterbox_image(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

