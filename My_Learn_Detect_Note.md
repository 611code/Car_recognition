# 车牌识别学习笔记

根据个人所编写的My_Learn_Detect.ipynb编写

## 包含必要的包

### 通用包

```python
import time
import os
import cv2
import torch
from numpy import random
import copy
import numpy as np
from matplotlib import pyplot as plt    # 导入pylot方法
```

### yolov5-5.0包

```python
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.torch_utils import time_synchronized
from utils.cv_puttext import cv2ImgAddText
```

### 个人所写包

```python
# 调试包
from Img_DeBug.Img_DeBug_Package import IMG_Debug
%matplotlib inline
```

## 建立通用变量

### 解释

```txt
plateName       车牌上要识别的字符
colors          车的颜色
clors           绘制时的颜色
object_color    未知
color_list      车牌的颜色
class_type      车牌的种类
danger          危险车牌的字符
mean_value      未知
std_value       未知
img_size        输入图像大小
conf_thres      置信度阈值
iou_thres       未知
dict_list       检测结果储存的字典
ratio_pad       图像缩放比例和填充的信息
device          选择模型所在设备运行
```

### 代码

```python
plateName=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
colors = ['黑色','蓝色','黄色','棕色','绿色','灰色','橙色','粉色','紫色','红色','白色']
clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
object_color=[(0,255,255),(0,255,0),(255,255,0)]
color_list=['黑色','蓝色','绿色','白色','黄色']
class_type=['单层车牌','双层车牌','汽车']
danger=['危','险']
mean_value,std_value=(0.588,0.193)
img_size = 384
conf_thres = 0.3
iou_thres = 0.5
dict_list=[]
ratio_pad = None

device = torch.device("cuda")
```

## 导入模型

### 解释

```txt
先导入模型
detect_model    车辆及车牌的识别模型
rec_model       车牌字符和颜色识别模型
car_rec_model   汽车颜色识别模型
detect_model
    attempt_load是yolov5官方提供的模型载入方式,也可以同时载入多个模型,暂未深入了解
plate_rec_model
    check_point为模型信息
```

### 代码

```python
from plate_recognition.plateNet import myNet_ocr,myNet_ocr_color
from car_recognition.myNet import myNet

detect_model = "weights/detect.pt"
rec_model = "weights/plate_rec_color.pth"
car_rec_model = "weights/car_rec_color.pth"

# 加载 detect_model 模型
detect_model = attempt_load(detect_model, map_location=device)  # load FP32 model
check_point = torch.load(rec_model,map_location=device)
model_state=check_point['state_dict']
cfg=check_point['cfg']
model_path = os.sep.join([sys.path[0],rec_model])
plate_rec_model = myNet_ocr_color(num_classes=len(plateName),export=True,cfg=cfg,color_num=len(color_list))
plate_rec_model.load_state_dict(model_state)
plate_rec_model.to(device)
plate_rec_model.eval()
# 加载car_rec_model模型
car_rec_model_check_point = torch.load(car_rec_model,map_location=torch.device(device))
cfg= car_rec_model_check_point['cfg']  
car_rec_model = myNet(num_classes=11,cfg=cfg)
car_rec_model.load_state_dict(car_rec_model_check_point['state_dict'])
car_rec_model.to(device) 
car_rec_model.eval()
```

### 输出

```
myNet(
  (feature): Sequential(
    (0): Conv2d(3, 8, kernel_size=(5, 5), stride=(1, 1))
    (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (4): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (8): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU(inplace=True)
    (11): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (12): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU(inplace=True)
    (15): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (16): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (18): ReLU(inplace=True)
  )
  (gap): AdaptiveAvgPool2d(output_size=(1, 1))
  (classifier): Linear(in_features=96, out_features=11, bias=True)
)
```

**print(check_point)** 的打印结果

```python
{
    'cfg': [8, 8, 16, 16, 'M', 32, 32, 'M', 48, 48, 'M', 64, 128], 
    'state_dict': OrderedDict(
        [
            (
                'feature.0.weight', 
                tensor(
                        [
                            [
                                [
                                    [ 2.96537e-02, -3.59661e-02,  8.16407e-02, -1.58546e-01,  5.47989e-02],
                                    [ 1.47463e-01, -1.12579e-01, -2.22962e-01, -3.30333e-01, -2.69785e-01],
                                    [-9.60916e-03,  5.68908e-02,  9.97541e-02, -1.59689e-01, -3.14875e-02],
                                    [ 1.37780e-01,  2.19792e-01,  1.33928e-01,  4.10988e-03, -4.32249e-03],
                                    [-3.73909e-02,  1.77340e-01,  1.30553e-01, -1.20739e-02, -1.64543e-01]
                                ],
                                [
                                    [-2.80289e-02, -2.18343e-01, -1.84176e-01, -5.00606e-01, -5.93086e-01],
                                    [ 2.28810e-01, -4.49530e-02, -1.21881e-01, -5.22211e-01, -6.42536e-01],
                                    [ 2.42228e-01,  2.86343e-01, -5.64169e-02, -2.26507e-01, -3.21826e-01],
                                    [ 3.17696e-01,  3.68173e-01,  2.25377e-01,  8.16352e-02, -2.98590e-01],
                                    [ 4.36584e-01,  4.50974e-01,  1.88365e-01,  1.18118e-01, -6.64403e-02]
                                ],
                                [
                                    [ 8.51759e-02, -4.46784e-02, -7.28180e-02, -2.34455e-01, -3.25261e-01],
                                    [ 1.83966e-01,  9.29289e-04, -1.54492e-01, -3.48359e-01, -2.68179e-01],
                                    [ 1.47569e-01,  1.97207e-01,  3.17762e-02, -1.54700e-01, -3.51396e-02],
                                    [ 2.60401e-01,  4.37296e-01,  3.46160e-01,  1.08465e-01, -1.44597e-01],
                                    [ 3.03784e-01,  2.50628e-01,  3.06603e-01,  2.77442e-01, -3.70473e-04]
                                ]
                            ],
                            [
                                [
                                    [ 1.60019e-01,  9.18172e-03, -4.23904e-02,  1.47494e-01,  3.06393e-02],
                                    [ 2.58340e-01, -1.28933e-02, -5.25636e-02, -5.63968e-02, -3.03716e-03],
                                    [ 5.94147e-02, -2.57058e-02, -1.77987e-01, -2.48552e-01, -1.08843e-01],
                                    [ 2.28385e-01, -1.16879e-01, -2.88118e-01, -2.51789e-01, -1.54778e-01],
                                    [ 1.02602e-01,  3.87202e-02, -1.93249e-01, -1.00373e-02,  7.95469e-03]
                                ],
                            ...
                                [
                                    [ 0.18941]
                                ]
                            ]
                        ]
                      )
            ), 
            (
                'newCnn.bias', 
                tensor(
                    [
                        -0.08999,  0.02127, -0.00681, -0.03790,  0.01267,  0.01637,  0.11666, -0.13731,  0.13195, -0.11415, -0.52541,  0.02719,  0.09062, -0.06359, -0.11064,  0.11710, -0.11194,  0.06451, -0.18159,  0.19102,  0.02919, -0.01809, -0.24531,  0.37997, -0.01935,  0.16260, -0.61637, -0.10610,  0.04478,  0.00628, -0.18955,0.07171, -0.46074, -0.13834, -0.38332, -0.35993, -0.45918,  0.06616, -0.44921, -0.00752, -0.47142, -0.23338,  0.36002, -0.00723,  0.16095,  0.11457,  0.00164,  0.05160,  0.16442,  0.16710,  0.25242, -0.10407, -0.09851,  0.03373, -0.11417, -0.12484, -0.08216, -0.11918, -0.06640,  0.08996, -0.05095, -0.12074,-0.05682, -0.10437,  0.10622,  0.19006, -0.19743,  0.12829, -0.05254, -0.15298, -0.49614, -0.05126, -0.20708,  0.06336,  0.03704, -0.31895, -0.07331, -0.17652
                    ]
                    )
            )
        ]
    )
}
```

## 载入图像

这里以单个图像为例

### 解释

**img 作为源图像**

### 代码

```python
# 载入图像
image_path = r"E:\01-Language\Python\PyTorch\Learning\Car_recognition\imgs\single_blue.jpg"
img = cv2.imread(image_path)
if img is None:
    raise ValueError("图像读取失败，请检查路径是否正确")
if img.shape[-1]==4:
    img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
IMG_Debug.PLT_Imgs([img],titles=["scoure"])
```

### 输出

![source](./image/Learn_Note/source.png)

## 图像填充

```txt
img0        为源图像复制而来，防止破坏源图像
h0, w0      图像尺寸
img_size    在上文有提到为384，作为统一输出的图像尺寸
r           r用来处理图像归一化，将最大边除以(或乘以)r得到标准图像尺寸
cv2.INTER_AREA      用于缩小图像
cv2.INTER_LINEAR    用于放大图像

实现效果：
    原图(372,500)
    过程为：r = 384/500=0.768
    r < 1,执行INTER_AREA
    将图像resize为(w0*r,h0*r) 即(285,384)
```

```python
if img.shape[-1]==4:
    img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
print("转化完成")
```

## 图像处理

### 代码

```python
img0 = copy.deepcopy(img)
assert img is not None, 'Image Not Found ' 
h0, w0 = img.shape[:2]  # orig hw
r = img_size / max(h0, w0)  # resize image to img_size
IMG_Debug.Data_DeBug(r,"比例")
IMG_Debug.Data_DeBug(img0.shape,"源图像的形状")
if r != 1:  # always resize down, only resize up if training with augmentation
    interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
    img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
IMG_Debug.Data_DeBug(img0.shape,"图像变换后的形状")
# check_img_size检查并调整图像尺寸，使其符合模型的要求。
# 获取模型要求的图像的最大的尺寸
imgsz = check_img_size(img_size, s=detect_model.stride.max())
IMG_Debug.Data_DeBug(imgsz,"图像的最大形状")
# 填充以符合模型步幅，可用color参数指定颜色
img_letterbox = letterbox(img0, new_shape=imgsz)[0]
IMG_Debug.Data_DeBug(img_letterbox.shape,"letterbox处理后的形状")
IMG_Debug.PLT_Imgs([img0,img_letterbox],titles=["img0","img_letterbox"])
```

### 输出

```python
[Message:0.3471971066907776, data:比例, shape:None, File:IMG_Debug.py, Line:148]
[Message:(826, 1106, 3), data:源图像的形状, shape:None, File:IMG_Debug.py, Line:148]
[Message:(286, 384, 3), data:图像变换后的形状, shape:None, File:IMG_Debug.py, Line:148]
[Message:384, data:图像的最大形状, shape:None, File:IMG_Debug.py, Line:148]
[Message:(320, 384, 3), data:letterbox处理后的形状, shape:None, File:IMG_Debug.py, Line:148]
```

![letterbox](./image/Learn_Note/letterbox.png)

## 张量处理

### 解释

效果：
    为了满足深度学习的图像框架
    将图像的形状(height, width, channels)转化为为(batch, channels, height, width)

#### 代码

```python
# 这行代码将图像的维度从 (height, width, channels) 调整为 (channels, height, width)。这是因为大多数深度学习框架
img_numpy = img_letterbox[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
IMG_Debug.Data_DeBug(img_numpy.shape,"ndarrary转化后的形状")
img_tensor = torch.from_numpy(img_numpy).to(device)
img_tensor = img_tensor.float()  # uint8 to fp16/32
# 数据归一化
img_tensor /= 255.0
# 若维度为3则增加一个维度
if img_tensor.ndimension() == 3:
    img_tensor = img_tensor.unsqueeze(0)
IMG_Debug.Data_DeBug(img_tensor.shape,"tensor转化后的形状")
```

#### 输出

```python
[Message:(3, 320, 384), data:ndarrary转化后的形状, shape:None, File:IMG_Debug.py, Line:148]
[Message:torch.Size([1, 3, 320, 384]), data:tensor转化后的形状, shape:None, File:IMG_Debug.py, Line:148]
```

## 汽车与车牌的位置预测

### 解释

```txt
下标为零表示获取图像的标注框坐标，类别，置信度，第一句未非极大值抑制，会得到非常多的预测框

使用python(pred.shape)可以得到框的数量
torch.Size([1, 7560, 16])   表示有7560个框
```

```python
tensor([[[6.78046e+00, 6.46662e+00, 1.20834e+01,  ..., 3.79657e-01, 8.29218e-02, 5.78657e-01],
         [1.41517e+01, 5.26659e+00, 1.48527e+01,  ..., 3.78371e-01, 9.89973e-02, 5.86137e-01],
         [2.07275e+01, 4.79271e+00, 1.54366e+01,  ..., 4.09637e-01, 1.51065e-01, 4.68648e-01],
         ...,
         [2.95318e+02, 2.90729e+02, 3.37368e+02,  ..., 1.74114e-02, 5.12017e-03, 9.81492e-01],
         [3.27860e+02, 2.93682e+02, 2.69169e+02,  ..., 1.03605e-02, 4.50116e-03, 9.89010e-01],
         [3.60833e+02, 2.93807e+02, 2.62076e+02,  ..., 1.18856e-02, 4.98215e-03, 9.87320e-01]]], device='cuda:0')
```

```txt
non_max_suppression_face(非极大值抑制处理后)
使用print(pred[0].shape)再次打印形状
torch.Size([4, 14])表示4个预测框，每个框有14个数据,这里以第一个为例
每个框的前4个数据表示坐标:       (76.28750, 58.30808) 到 (298.29974, 272.32562)
置信度分数：                    0.93487
关键点坐标：                    (134.74045, 159.12534), (229.89737, 161.36128), (228.88342, 201.55840), (133.69955, 199.95354)
类别标签：2.00000
```

```python
print(pred)
[tensor([[ 76.28750,  58.30808, 298.29974, 272.32562,   0.93487, 134.74045, 159.12534, 229.89737, 161.36128, 228.88342, 201.55840, 133.69955, 199.95354,   2.00000],
        [163.23050, 194.32040, 221.69077, 224.77530,   0.85369, 163.77687, 194.50372, 220.55659, 194.87260, 220.86121, 224.32050, 164.15297, 224.00162,   1.00000],
        [328.28217, 116.52200, 383.88123, 168.85281,   0.69716, 336.82529, 128.96635, 375.73248, 127.13684, 373.25079, 152.56892, 336.94781, 153.89563,   2.00000],
        [ 13.83384, 106.86729,  84.16957, 137.81703,   0.42462,  31.67604, 117.09824,  70.34735, 118.78855,  70.01311, 129.93410,  30.54813, 128.33952,   2.00000]], device='cuda:0')]
```

### 代码

```python
# 预测的图像的img_tensor不是源图像，根据预测结果要变换坐标到原图上去
pred = detect_model(img_tensor)[0]
IMG_Debug.Data_DeBug(pred.shape,"预测框数据的形状(非极大值阈值处理前)")
pred = non_max_suppression_face(pred, conf_thres, iou_thres)
IMG_Debug.Data_DeBug(pred[0].shape,"预测框数据的形状(非极大值阈值处理后)")
print(pred)
```

### 输出

```txt
[Message:torch.Size([1, 7560, 16]), data:预测框数据的形状(非极大值阈值处理前), shape:None, File:IMG_Debug.py, Line:148]
[Message:torch.Size([4, 14]), data:预测框数据的形状(非极大值阈值处理后), shape:None, File:IMG_Debug.py, Line:148]
[tensor([[ 79.91327, 103.28557, 324.76053, 241.94810,   0.90371, 141.40739, 160.62038, 261.88049, 152.69121, 266.59634, 194.40169, 147.13666, 204.90102,   2.00000],
        [ 83.33043, 193.95425, 103.84804, 216.36282,   0.88346,  84.48236, 193.97292, 103.29549, 202.34676, 102.59531, 216.29401,  83.81242, 207.86284,   0.00000],
        [333.33786, 142.91516, 352.25418, 150.25293,   0.82946, 333.50650, 142.98677, 352.00833, 143.92868, 351.98047, 150.26184, 333.49014, 149.31726,   0.00000],
        [319.39932,  98.59377, 384.58060, 159.46361,   0.81391, 344.03732, 133.23834, 363.57947, 135.68257, 362.89606, 149.77693, 342.27228, 147.51096,   2.00000]], device='cuda:0')]
```

## 汽车与车牌颜色及字符识别

下面将详细解释代码，建议结合Learn_Note.ipynb来看，这里先把完整代码列出

#### 代码

```python
rects = []
rects.append(img)
# 依次遍历预测框
for i,det in enumerate(pred):
    # 检测有没有预测到东西
    if len(det):
        # 检测是否已经提供了图像缩放比例和填充的信息，若没有则按照计算的进行
        if ratio_pad is None:  # calculate from img0_shape
            # 缩放比例（gain）是通过计算变换后图像张量的宽度和高度与原始图像的宽度和高度的比例得到的。
            gain = min(img_tensor.shape[2:][0] / img.shape[0],img_tensor.shape[2:][1] / img.shape[1])  # gain
            print(gain)
            # 填充量（pad）是通过计算变换后图像张量的宽度和高度与缩放后的原始图像的宽度和高度之间的差值的一半得到的。
            pad = (img_tensor.shape[2:][1] - img.shape[1] * gain) / 2, (img_tensor.shape[2:][0] - img.shape[0] * gain) / 2
            print("pad",img_tensor.shape[2:][1],img.shape[1],img.shape[1] * gain)
            print("pad",img_tensor.shape[2:][0],img.shape[0],img.shape[0] * gain)
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        # det[:,:4] 表示选中每个框的数据(:),每个框的前四个数据(:4),为每个预测框的前四个数据，表示框的坐标
        # det[:,:4][:, [0, 2]] 表示每个预测框的数据的前四个数据的第0列和第2列([0,2]),即左上角和右下角的x坐标
        # 变换坐标得到原图的预测框的坐标
        det[:, :4][:, [0, 2]] -= pad[0]  # x padding
        det[:, :4][:, [1, 3]] -= pad[1]  # y padding
        det[:, :4][:, :4] /= gain
        # 将该坐标限制在源图像里，防止图像框的坐标为负，在图像外边.超过最大值，变为最大值，小于最小值，变为0
        det[:, :4][:, 0].clamp_(0, img.shape[1])  # x1
        det[:, :4][:, 1].clamp_(0, img.shape[0])  # y1
        det[:, :4][:, 2].clamp_(0, img.shape[1])  # x2
        det[:, :4][:, 3].clamp_(0, img.shape[0])  # y2
  
        # 它遍历所有不同的类别，并计算每个类别在检测结果中出现的次数。
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
        print("n",n)
        # print(det)
        # if ratio_pad is None:  # calculate from img0_shape
        #     gain = min(img_tensor.shape[2:][0] / img.shape[0], img_tensor.shape[2:][1] / img.shape[1])  # gain  = old / new
        #     pad = (img_tensor.shape[2:][1] - img.shape[1] * gain) / 2, (img_tensor.shape[2:][0] - img.shape[0] * gain) / 2  # wh padding
        # else:
        #     gain = ratio_pad[0][0]
        #     pad = ratio_pad[1]
        # 同上，变换关键点坐标
        det[:, 5:13][:, [0, 2, 4, 6]] -= pad[0]  # x padding
        det[:, 5:13][:, [1, 3, 5, 7]] -= pad[1]  # y padding
        det[:, 5:13][:, :8] /= gain
        # 同上，限制关键点坐标
        det[:, 5:13][:, 0].clamp_(0, img.shape[1])  # x1
        det[:, 5:13][:, 1].clamp_(0, img.shape[0])  # y1
        det[:, 5:13][:, 2].clamp_(0, img.shape[1])  # x2
        det[:, 5:13][:, 3].clamp_(0, img.shape[0])  # y2
        det[:, 5:13][:, 4].clamp_(0, img.shape[1])  # x3
        det[:, 5:13][:, 5].clamp_(0, img.shape[0])  # y3
        det[:, 5:13][:, 6].clamp_(0, img.shape[1])  # x4
        det[:, 5:13][:, 7].clamp_(0, img.shape[0])  # y4
        # 以上仅为预测框中的坐标变换
        print("det.size()[0]",det.size())
        # det.size() 为 torch.Size([4, 14])
        for j in range(det.size()[0]):
            # view(-1) 用于将张量展平成一维张量；tolist()将张量转化为列表
            # 获取第j个预测框中的前4个数据,展成1维,转为列表
            xyxy = det[j, :4].view(-1).tolist()
            conf = det[j, 4].cpu().numpy()
            landmarks = det[j, 5:13].view(-1).tolist()
            class_num = det[j, 13].cpu().numpy()
            # img_test2 = img[:,:]
            # img_test2 = cv2.circle(img_test2, (407,414), 10, (0,255,0), 13)
            # img_test2 = cv2.circle(img_test2, (754,754), 10, (0,255,0), 13)
            # img_test2 = cv2.circle(img_test2, (767,512), 10, (0,255,0), 13)
            # img_test2 = cv2.circle(img_test2, (423,542), 10, (0,255,0), 13)
            # rects.append(img_test2)
            """
            print("landmarks",landmarks)
            [407.2827453613281, 414.78680419921875, 754.2703857421875, 754.94915771484375, 767.8529663085938, 512.0840454101562, 423.7842102050781, 542.3242797851562]
            [243.32679748535156, 510.8490905761719, 297.5125427246094, 534.9674682617188, 295.4958801269531, 575.1384887695312, 241.39723205566406, 550.8549194335938]
            [960.5681762695312, 363.99835205078125, 1013.8572998046875, 366.71124267578125, 1013.7770385742188, 384.95208740234375, 960.5210571289062, 382.2314758300781]
            [990.8991088867188, 335.92083740234375, 1047.1845703125, 342.9607238769531, 1045.2161865234375, 383.5554504394531, 985.8154296875, 377.0289611816406]
            """
            # 此次的循环为一次循环,目的在于检测到指定的物体时可以直接跳出来,而不用再把其他物体再检测一遍
            for i in range(1):
                # 获取图像的高度,宽度,通道
                h,w,c = img.shape
                # 保存识别数据
                result_dict={}
                # 获取预测图坐标
                x1 = int(xyxy[0])
                y1 = int(xyxy[1])
                x2 = int(xyxy[2])
                y2 = int(xyxy[3])
                landmarks_np=np.zeros((4,2))
                rect=[x1,y1,x2,y2]
                # 检测是否为车辆
                if int(class_num) == 2:
                    car_roi_img = img[y1:y2,x1:x2]
                    # 用于查看图像
                    rects.append(car_roi_img)
                    img_car_rec = cv2.resize(car_roi_img,(64,64))
                    rects.append(img_car_rec)
                    # transpose 方法用于重新排列数组的维度顺序。
                    img_car_rec = img_car_rec.transpose([2,0,1])
                    img_car_rec = torch.from_numpy(img_car_rec).float().to(device)
                    # 为了将像素值从 [0, 255] 范围归一化到 [-127.5, 127.5] 范围
                    img_car_rec = img_car_rec-127.5
                    # unsqueeze 方法用于在张量的指定位置插入一个新的维度（轴）
                    img_car_rec = img_car_rec.unsqueeze(0)
                    # 检测车辆颜色
                    result = car_rec_model(img_car_rec)
                    """ 
                    print("result",result)
                    tensor([[-0.70725, -0.56493,  0.02660, -0.72722, -0.50298, -0.03824, -0.39895, -0.69850, -0.61010, -0.00711,  4.22289]], device='cuda:0', grad_fn=<AddmmBackward0>)
                    tensor([[ 0.54220, -0.42025, -0.35115, -0.77246, -1.30125,  0.89859, -0.69436, -0.52382, -0.82337, -0.89630,  4.34603]], device='cuda:0', grad_fn=<AddmmBackward0>)
                    tensor([[ 0.59463, -0.36 571, -0.62544, -1.00333, -0.62139,  2.94423, -0.70958, -0.89029, -0.92662, -0.60373,  2.20735]], device='cuda:0', grad_fn=<AddmmBackward0>)
                    每个张量都包含一组浮点数值，并且这些张量位于 GPU 上（device='cuda:0'）。
                    每个张量都有一个 grad_fn=<AddmmBackward0> 属性，表明这些张量是由某个涉及矩阵乘法的操作生成的，并且它们带有梯度信息，通常用于反向传播计算。
                    """
                    # Softmax 函数常用于将一个向量转换为概率分布，使得所有元素的和为 1。
                    out =F.softmax( result)
                    """
                    print("out",out)
                    tensor([[0.00657, 0.00757, 0.01368, 0.00644, 0.00806, 0.01282, 0.00894, 0.00662, 0.00724, 0.01323, 0.90884]], device='cuda:0', grad_fn=<SoftmaxBackward0>)
                    tensor([[0.02014, 0.00769, 0.00824, 0.00541, 0.00319, 0.02877, 0.00585, 0.00694, 0.00514, 0.00478, 0.90385]], device='cuda:0', grad_fn=<SoftmaxBackward0>)
                    tensor([[0.05350, 0.02048, 0.01579, 0.01082, 0.01586, 0.56073, 0.01452, 0.01212, 0.01169, 0.01614, 0.26836]], device='cuda:0', grad_fn=<SoftmaxBackward0>)
                    """
                    _, predicted = torch.max(out.data, 1)
                    """
                    print(_,predicted)
                    tensor([0.90884], device='cuda:0') tensor([10], device='cuda:0')
                    tensor([0.90385], device='cuda:0') tensor([10], device='cuda:0')
                    tensor([0.56073], device='cuda:0') tensor([5], device='cuda:0')
                    """
                    out=out.data.cpu().numpy().tolist()[0]
                    """
                    print(out,"out")
                    [0.006566870026290417, 0.007571233436465263, 0.01367923803627491 , 0.006436999887228012, 0.008055126294493675 , 0.012820458970963955, 0.008938240818679333, 0.006624543573707342, 0.007236848119646311, 0.013225853443145752, 0.9088444709777832 ]
                    [0.020142601802945137, 0.007693612016737461, 0.008244002237915993, 0.005409614648669958, 0.0031879874877631664, 0.02876695804297924 , 0.005849051754921675, 0.006936600431799889, 0.005141084548085928, 0.004779500886797905, 0.9038490653038025 ]
                    [0.053497541695833206, 0.02047671191394329 , 0.015792904421687126, 0.01082293875515461 , 0.015856925398111343 , 0.56072598695755    , 0.0145184975117445  , 0.01211819238960743 , 0.01168590597808361 , 0.01613946631550789 , 0.26836487650871277]
                    """
                    predicted = predicted.item()
                    """
                    print("predicted",predicted)
                    10
                    10
                    5
                    """
                    car_color= colors[predicted]
                    """
                    print("car_color",car_color)
                    白色
                    白色
                    灰色
                    """
                    color_conf = out[predicted]
                    """ 
                    print("color_conf",color_conf)
                    color_conf 0.9088444709777832
                    color_conf 0.9038490653038025
                    color_conf 0.56072598695755 
                    """
                    result_dict['class_type']=class_type[int(class_num)]
                    result_dict['rect']=rect                      #车辆roi
                    result_dict['score']=conf                     #车牌区域检测得分
                    result_dict['object_no']=int(class_num)
                    result_dict['car_color']=car_color
                    result_dict['color_conf']=color_conf
                    dict_list.append(result_dict)
                    break
                for i in range(4):
                    # 第一个点的坐标的x,y
                    point_x = int(landmarks[2 * i])
                    point_y = int(landmarks[2 * i + 1])
                    # 转化为ndarray 储存到 landmarks_np
                    landmarks_np[i]=np.array([point_x,point_y])
  
                """ 
                print("landmarks_np",landmarks_np)
                [[        213         230]
                [        287         231]
                [        287         269]
                [        213         269]] 
                """
                class_label= int(class_num)
                rect1 = np.zeros((4, 2), dtype = "float32")  # 4 行 2 列。
                # 将landmarks_np的第一维求和
                s = landmarks_np.sum(axis = 1)
                """ 
                print("s",s)
                s [        443         518         556         482] 
                """
                # argmin() 方法用于找到数组或张量中最小值的索引,即左上角
                rect1[0] = landmarks_np[np.argmin(s)]
                # argmax() 方法用于找到数组或张量中最大值的索引,即右下角
                rect1[2] = landmarks_np[np.argmax(s)]
                # diff求差，argmin为小，即
                diff = np.diff(landmarks_np, axis = 1)
                # 寻找差最大和最小的,即宽和高
                rect1[1] = landmarks_np[np.argmin(diff)]
                rect1[3] = landmarks_np[np.argmax(diff)]
                """
                print("rect1",rect1,"diff",diff)
                [
                    [        213         230]
                    [        287         231]
                    [        287         269]
                    [        213         269]
                ]
                [
                    [         17]
                    [        -56]
                    [        -18]
                    [         56]
                ]
                """
                (tl, tr, br, bl) = rect1
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")
                print("test",rect1,dst)
                # imgtest = img[:]
                # imgtest = cv2.circle(imgtest, (243,510), 5, (0,255,0), 10)
                # imgtest = cv2.circle(imgtest, (299,534), 5, (0,255,0), 10)
                # imgtest = cv2.circle(imgtest, (241,550), 5, (0,255,0), 10)
                # imgtest = cv2.circle(imgtest, (295,575), 5, (0,255,0), 10)
                # rects.append(imgtest)
  
                M = cv2.getPerspectiveTransform(rect1, dst)
                # 该图像 roi_img 为 车牌的图像
                roi_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
                rects.append(roi_img)
                # plt_showBGR("car_plate",roi_img)
                if class_label:                                                             #判断是否是双层车牌，是双牌的话进行分割后然后拼接
                    # roi_img=get_split_merge(roi_img)
                    h,w,c = roi_img.shape
                    img_upper = roi_img[0:int(5/12*h),:]
                    img_lower = roi_img[int(1/3*h):,:]
                    img_upper = cv2.resize(img_upper,(img_lower.shape[1],img_lower.shape[0]))
                    roi_img = np.hstack((img_upper,img_lower))
                # plate_number ,plate_color= get_plate_result(roi_img,device,plate_rec_model)
                roi_img = cv2.resize(roi_img, (168,48))
                roi_img = np.reshape(roi_img, (48, 168, 3))

                roi_img = roi_img.astype(np.float32)
                roi_img = (roi_img / 255. - mean_value) / std_value
                roi_img = roi_img.transpose([2, 0, 1])
                roi_img = torch.from_numpy(roi_img)

                roi_img = roi_img.to(device)
                roi_img = roi_img.view(1, *roi_img.size())
  
                preds,color_preds = plate_rec_model(roi_img)
                preds =preds.argmax(dim=2) #找出概率最大的那个字符
                color_preds = color_preds.argmax(dim=-1)
                # print(preds.tolist()[0])
                # for zi in preds.tolist()[0]:
                #   # print(plateName[zi])
                preds=preds.view(-1).detach().cpu().numpy()
                color_preds=color_preds.item()
                # newPreds=decodePlate(preds)
                pre=0
                newPreds=[]
                for i in range(len(preds)):
                    if preds[i]!=0 and preds[i]!=pre:
                        newPreds.append(preds[i])
                    pre=preds[i]
                plate=""
                for i in newPreds:
                    plate+=plateName[i]
                plate_number ,plate_color = plate,color_list[color_preds]
              # print(plate_number ,plate_color)
                for dan in danger:                                                          #只要出现‘危’或者‘险’就是危险品车牌
                    if dan in plate_number:
                        plate_number='危险品'
                result_dict['class_type']=class_type[class_label]
                result_dict['rect']=rect                            #车牌roi区域
                # print(rect)
                result_dict['landmarks']=landmarks_np.tolist()      #车牌角点坐标
                result_dict['plate_no']=plate_number                #车牌号
                result_dict['roi_height']=roi_img.shape[0]          #车牌高度
                result_dict['plate_color']=plate_color              #车牌颜色
                result_dict['object_no']=class_label                #单双层 0单层 1双层
                result_dict['score']=conf                           #车牌区域检测得分
                dict_list.append(result_dict)
IMG_Debug.PLT_Imgs(rects,lines=5)
```

### 代码分立讲解

#### 放缩与填充

```python
for i,det in enumerate(pred):
    if len(det):
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img_tensor.shape[2:][0] / img.shape[0],img_tensor.shape[2:][1] / img.shape[1])  # gain
            print(img_tensor.shape[2:][0],img.shape[0],img_tensor.shape[2:][1],img.shape[1],img_tensor.shape[2:][0] / img.shape[0],img_tensor.shape[2:][1] / img.shape[1],gain)
            pad = (img_tensor.shape[2:][1] - img.shape[1] * gain) / 2, (img_tensor.shape[2:][0] - img.shape[0] * gain) / 2
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
```

#### 输出

```python
320 826 384 1106 0.387409200968523 0.3471971066907776 0.3471971066907776
```

#### 解释

```txt
for i,det in enumerate(pred):
    遍历每个预测框
if len(det):
    如果det存在，则进入该条件语句
if ratio_pad is None:
    如果 ratio_pad 没定义，则按照下面的语句计算 放缩比例gain 和 填充量pad，否则按照定义的 放缩比例和填充量
gain = min(img_tensor.shape[2:][0] / img.shape[0],img_tensor.shape[2:][1] / img.shape[1])  # gain
    使用变换后的图像的宽除以源图像的宽，使用变换后的图像的高除以源图像的高，取它们的最小值
```

#### 关键点变换

```python
det[:, :4][:, [0, 2]] -= pad[0]  # x padding
det[:, :4][:, [1, 3]] -= pad[1]  # y padding
det[:, :4][:, :4] /= gain
print("det",det[:, :4])
# 将该坐标限制在源图像里，防止图像框的坐标为负，在图像外边.超过最大值，变为最大值，小于小值，变为0
det[:, :4][:, 0].clamp_(0, img.shape[1])  # x1
det[:, :4][:, 1].clamp_(0, img.shape[0])  # y1
det[:, :4][:, 2].clamp_(0, img.shape[1])  # x2
det[:, :4][:, 3].clamp_(0, img.shape[0])  # y2

det[:, 5:13][:, [0, 2, 4, 6]] -= pad[0]  # x padding
det[:, 5:13][:, [1, 3, 5, 7]] -= pad[1]  # y padding
det[:, 5:13][:, :8] /= gain
# 同上，限制关键点坐标
det[:, 5:13][:, 0].clamp_(0, img.shape[1])  # x1
det[:, 5:13][:, 1].clamp_(0, img.shape[0])  # y1
det[:, 5:13][:, 2].clamp_(0, img.shape[1])  # x2
det[:, 5:13][:, 3].clamp_(0, img.shape[0])  # y2
det[:, 5:13][:, 4].clamp_(0, img.shape[1])  # x3
det[:, 5:13][:, 5].clamp_(0, img.shape[0])  # y3
det[:, 5:13][:, 6].clamp_(0, img.shape[1])  # x4
det[:, 5:13][:, 7].clamp_(0, img.shape[0])  # y4

print("det[:, 5:13]",det[:, 5:13])
```

#### 输出

```python
det tensor([[ 230.16685,  249.65060,  935.37793,  649.02759],
        [ 240.00899,  510.79532,  299.10397,  575.33667],
        [ 960.08246,  363.79208, 1014.56543,  384.92642],
        [ 919.93658,  236.13724, 1107.67224,  411.45508]], device='cuda:0')
```

这里的 det[:,:4] 输出即为源图像的物体的坐标(左上x,y,右下x,y)
![物体位置](./image/Learn_Note/object_xy_rect.png)
clamp_ 用来限制数据范围(0, img.shape[n])

```python
det[:, 5:13] tensor([[ 407.28275,  414.78680,  754.27039,  391.94916,  767.85297,  512.08405,  423.78421,  542.32428],
        [ 243.32680,  510.84909,  297.51254,  534.96747,  295.49588,  575.13849,  241.39723,  550.85492],
        [ 960.56818,  363.99835, 1013.85730,  366.71124, 1013.77704,  384.95209,  960.52106,  382.23148],
        [ 990.89911,  335.92084, 1047.18457,  342.96072, 1045.21619,  383.55545,  985.81543,  377.02896]], device='cuda:0')
```

det[:, 5:13] 为物体的4个关键点的坐标，这里主要用于车牌
![车牌关键点](./image/Learn_Note/car_plate_keypoints.png)

#### 解释

```txt
这里的det是根据letterbox填充后的图像来判断的，与源图像不同，需要进行一系列的变换

det[:, :4]              表示每个预测框的0-4位数据,即预测的物体的坐标(左上角x,y,右上角x,y)
det[:, :4][:, [0, 2]]   表示每个预测框的0-4位数据的第0位与第2位的数据，即两个坐标的x量
-= pad[0]               是为了减掉letterbox的多余的填充，得到了填充后的相对于无填充的相对坐标
/= gain                 除以放缩比例，得到了源图像的物体坐标数据

```

#### 车牌坐标变换

```python
for j in range(det.size()[0]):
    xyxy = det[j, :4].view(-1).tolist()
    conf = det[j, 4].cpu().numpy()
    landmarks = det[j, 5:13].view(-1).tolist()
    class_num = det[j, 13].cpu().numpy()

```

#### 解释

| 代码                           | 意义                                  |
| ------------------------------ | ------------------------------------- |
| for j in range(det.size()[0]): | 遍历预测框的数量                      |
| size方法                       | 获取det的形状，[0]取det的预测框的数量 |
| xyxy                           | 物体的坐标                            |
| conf                           | 物体的置信度                          |
| landmarks                      | 物体关键点的坐标                      |
| class_num                      | 物体的种类                            |

#### 变量准备

```python
for i in range(1):
    h,w,c = img.shape
    result_dict={}
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    landmarks_np=np.zeros((4,2))
    rect=[x1,y1,x2,y2]
```

#### 解释

| 代码               | 意义                          |
| ------------------ | ----------------------------- |
| for i in range(1): | 循环一次是为了随时(break)退出 |
| result_dict={}     | 将检测结果保存到此处          |

#### 车辆处理

```python
if int(class_num) ==2:
    car_roi_img = img[y1:y2,x1:x2]
    img_car_rec = cv2.resize(car_roi_img,(64,64))
    img_car_rec = img_car_rec.transpose([2,0,1])
    img_car_rec = torch.from_numpy(img_car_rec).float().to(device)
    img_car_rec = img_car_rec-127.5
    img_car_rec = img_car_rec.unsqueeze(0)
    result = car_rec_model(img_car_rec)
    out =F.softmax( result)
    _, predicted = torch.max(out.data, 1)
    out=out.data.cpu().numpy().tolist()[0]
    predicted = predicted.item()
    car_color= colors[predicted]
    color_conf = out[predicted]
    result_dict['class_type']=class_type[int(class_num)]
    result_dict['rect']=rect                      #车辆roi
    result_dict['score']=conf                     #车牌区域检测得分
    result_dict['object_no']=int(class_num)
    result_dict['car_color']=car_color
    result_dict['color_conf']=color_conf
    dict_list.append(result_dict)
    # print(result_dict)
    break
```

#### 解释

| 代码                                                           | 意义                                                                                                   |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| if int(class_num) ==2:                                         | 判断是否为汽车                                                                                         |
| car_roi_img = img[y1:y2,x1:x2]                                 | 获取源图像中车辆的部分                                                                                 |
| img_car_rec = cv2.resize(car_roi_img,(64,64))                  | 调整尺寸                                                                                               |
| img_car_rec = img_car_rec.transpose([2,0,1])                   | 更改形状顺序                                                                                           |
| img_car_rec = torch.from_numpy(img_car_rec).float().to(device) | 转为tensor变量                                                                                         |
| img_car_rec = img_car_rec-127.5                                | 将范围限制在-127.5-127.5                                                                               |
| img_car_rec = img_car_rec.unsqueeze(0)                         | 插入一个维度                                                                                           |
| result = car_rec_model(img_car_rec)                            | 预测汽车颜色                                                                                           |
| out =F.softmax( result)                                        | 它将一个 K 维的向量转换为另一个 K 维的向量，<br />使得每个元素都在 0 和 1 之间，并且所有元素之和等于 1 |
| _, predicted = torch.max(out.data, 1)                          | 返回第一个维度的最大数据的索引                                                                         |
| out=out.data.cpu().numpy().tolist()[0]                         | 将out的数据转移到cpu中，并通过转为numpy后转为tolist())                                                 |
| predicted = predicted.item()                                   | 将 predicted 转为int                                                                                  |
| car_color= colors[predicted]                                   | predicted所对应的颜色                                                                                  |
| color_conf = out[predicted]                                    | predicted所对应的置信度                                                                                |
| dict_list.append(result_dict)                                  | 把结果存储在dict_list                                                                                  |

#### 车牌图像信息获取

```python
for i in range(4):
    point_x = int(landmarks[2 * i])
    point_y = int(landmarks[2 * i + 1])
    landmarks_np[i]=np.array([point_x,point_y])
rect1 = np.zeros((4, 2), dtype = "float32")  # 4 行 2 列。
s = landmarks_np.sum(axis = 1)
rect1[0] = landmarks_np[np.argmin(s)]
rect1[2] = landmarks_np[np.argmax(s)]
# diff求差，argmin为小，即
diff = np.diff(landmarks_np, axis = 1)
rect1[1] = landmarks_np[np.argmin(diff)]
rect1[3] = landmarks_np[np.argmax(diff)]
```

#### 解释

| 代码                                                                                   | 意义                                                                                          |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| ![1732961108866](image/Learn_Note/1732961108866.png)                                     | 获取landmarks的数据（点的坐标）<br />代码执行后landmarks_np为一个储存了4个关键点坐标的ndarray |
| s = landmarks_np.sum(axis = 1)                                                         | 计算这4个坐标的和，以此判断左上角和右下角，或者宽高                                           |
| rect1[0]= landmarks_np[np.argmin(s)]<br />rect1[2]= landmarks_np[np.argmax(s)]         | 根据s的最小值和最大值的索引，判断左上角和右下角                                               |
| diff = np.diff(landmarks_np, axis = 1)                                                 | 对landmarks_np的第一个维度做差，用来获取高和宽                                                |
| rect1[1] = landmarks_np[np.argmin(diff)]<br />rect1[3] = landmarks_np[np.argmax(diff)] | 根据s的最小值和最大值的索引，判断右上角和左下角                                               |

这段代码执行完后rect1为车牌的左上，右上，右下，左下

#### 车牌坐标变换

```python
(tl, tr, br, bl) = rect1
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))
dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
M = cv2.getPerspectiveTransform(rect1, dst)
roi_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
```

#### 解释

| 代码                                                              | 意义                                                    |
| ----------------------------------------------------------------- | ------------------------------------------------------- |
| widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)) | 计算两点之间的欧几里得距离                              |
| ![1732962693210](image/Learn_Note/1732962693210.png)                | 分别对应左上，右上，右下，左下                          |
| M = cv2.getPerspectiveTransform(rect1, dst)                       | 将rect1变换到dst的方法                                  |
| roi_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))      | 将M方法应用与img，(maxWidth, maxHeight)为输出的图像大小 |

输出的图像

![car_plate_Transform](./image/Learn_Note/car_plate_Transform.png)

#### 双层车牌处理

```python
if class_label: #判断是否是双层车牌，是双牌的话进行分割后然后拼接
    h,w,c = roi_img.shape
    img_upper = roi_img[0:int(5/12*h),:]
    img_lower = roi_img[int(1/3*h):,:]
    img_upper = cv2.resize(img_upper,(img_lower.shape[1]img_lower.shape[0]))
    roi_img = np.hstack((img_upper,img_lower))
```

#### 解释

最下为效果，当前案例没有双层车牌，用的其他图片

| code                                                                     | 意义                                     |
| ------------------------------------------------------------------------ | ---------------------------------------- |
| if class_label:                                                          | class_label表示单双层 ：0单层 1双层      |
| h,w,c = roi_img.shape                                                    | 获取车牌图像形状信息                     |
| img_upper = roi_img[0:int(5/12*h),:]                                     | 获取整个图像从上到下0~5/12的部分         |
| img_upper = cv2.resize(img_upper,(img_lower.shape[1]img_lower.shape[0])) | 改变上层的图像的尺寸以能与下层图像拼接   |
| img_lower = roi_img[int(1/3*h):,:]                                       | 获取整个图像从上到下1/3~1的部分          |
| roi_img = np.hstack((img_upper,img_lower))                               | 将其水平拼接                             |
| ![double1](./image/Learn_Note/double1.png)                                 | ![double2](./image/Learn_Note/double2.png) |

#### 车牌字符识别

```python
roi_img = cv2.resize(roi_img, (168,48))
roi_img = np.reshape(roi_img, (48, 168, 3))
roi_img = roi_img.astype(np.float32)
roi_img = (roi_img / 255. - mean_value) / std_value
roi_img = roi_img.transpose([2, 0, 1])
roi_img = torch.from_numpy(roi_img)
roi_img = roi_img.to(device)
roi_img = roi_img.view(1, *roi_img.size())

preds,color_preds = plate_rec_model(roi_img)
preds =preds.argmax(dim=2) #找出概率最大的那个字符
color_preds = color_preds.argmax(dim=-1)
# print(preds.tolist()[0])
# for zi in preds.tolist()[0]:
#   # print(plateName[zi])
preds=preds.view(-1).detach().cpu().numpy()
color_preds=color_preds.item()
```

#### 解释

| 代码                                                | 解释                                                                                                                                                                                 |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| roi_img = cv2.resize(roi_img, (168,48))             | 改变尺寸，适应深度学习框架                                                                                                                                                           |
| roi_img = np.reshape(roi_img, (48, 168, 3))         | 改变通道顺序                                                                                                                                                                         |
| roi_img = roi_img.astype(np.float32)                | 转为浮点型，以归一化                                                                                                                                                                 |
| roi_img = (roi_img / 255. - mean_value) / std_value | mean_value均值通常是通过对训练数据集中的所有图像计算得到的，<br />表示图像的平均亮度<br />std_value 标准差也是通过对训练数据集中的所有图像计算得到的，<br />表示图像的亮度变化程度。 |
| roi_img = roi_img.transpose([2, 0, 1])              | 更改通道顺序                                                                                                                                                                         |
| roi_img = torch.from_numpy(roi_img)                 | 转为tensor类型                                                                                                                                                                       |
| roi_img = roi_img.to(device)                        | 放到指定设备上预测                                                                                                                                                                   |
| roi_img = roi_img.view(1, *roi_img.size())          | 改变张量形状                                                                                                                                                                         |
| preds,color_preds =plate_rec_model(roi_img)         | 得到预测结果，preds为预测的字符概率，color_preds为预测的颜色概率                                                                                                                     |
| preds =preds.argmax(dim=2)                          | 找出第二个维度中概率最大的那个字符                                                                                                                                                   |
| preds=preds.view(-1).detach().cpu().numpy()         | detach()创建一个不会跟踪的张量                                                                                                                                                       |
| color_preds=color_preds.item()                      | 转化为int类型                                                                                                                                                                        |

#### 车牌字符数据处理

```python
pre=0
newPreds=[]
for i in range(len(preds)):
    if preds[i]!=0 and preds[i]!=pre:
        newPreds.append(preds[i])
    pre=preds[i]
plate=""
for i in newPreds:
    plate+=plateName[i]
plate_number ,plate_color = plate,color_list[color_preds]
for dan in danger:                                                          #只要出现‘危’或者‘险’就是危品车牌
    if dan in plate_number:
        plate_number='危险品'
result_dict['landmarks']=landmarks_np.tolist()      #车牌角点坐标
result_dict['plate_no']=plate_number                #车牌号
result_dict['roi_height']=roi_img.shape[0]          #车牌高度
result_dict['plate_color']=plate_color              #车牌颜色
result_dict['object_no']=class_label                #单双层 0单层 1双层
result_dict['score']=conf                           #车牌区域检测得分
dict_list.append(result_dict)
```

#### 解释

preds： **[12  0  0  0 53  0  0 44  0  0 71  0 51 51  0  0 62  0  0 49 49]**

| 代码                                               | 意义                                                            |
| -------------------------------------------------- | --------------------------------------------------------------- |
| pre=0<br />newPreds=[]                             | 前提变量暂存信息                                                |
| for i inrange(len(preds)):                         | 遍历预测的信息的长度                                            |
| if preds[i]!=0 and preds[i]!=pre:                  | 如果预测的字符对应的下标不等于0同时不等于前一个时，视为有效字符 |
| newPreds.append(preds[i])                          | 存储到newPreds                                                  |
| pre=preds[i]                                       | 将pre等于当前结果，用于下次预测                                 |
| ![1732969855067](image/Learn_Note/1732969855067.png) | 如果车牌中含有'危'或'险',即为危险品车牌                         |
| ![1732969960547](image/Learn_Note/1732969960547.png) | 存放识别信息到dict_list                                         |

**dict_list:**

[{'class_type': '汽车', 'rect': [230, 249, 935, 649], 'score': array(    0.90371, dtype=float32), 'object_no': 2, 'car_color': '红色', 'color_conf': 0.652275800704956}, {'class_type': '单层车牌', 'rect': [240, 510, 299, 575], 'landmarks': [[243.0, 510.0], [297.0, 534.0], [295.0, 575.0], [241.0, 550.0]], 'plate_no': '浙B2V9L7', 'roi_height': 1, 'plate_color': '蓝色', 'object_no': 0, 'score': array(    0.88346, dtype=float32)}, {'class_type': '单层车牌', 'rect': [960, 363, 1014, 384], 'landmarks': [[960.0, 363.0], [1013.0, 366.0], [1013.0, 384.0], [960.0, 382.0]], 'plate_no': '辽DU4356', 'roi_height': 1, 'plate_color': '蓝色', 'object_no': 0, 'score': array(    0.82946, dtype=float32)}, {'class_type': '汽车', 'rect': [919, 236, 1106, 411], 'score': array(    0.81391, dtype=float32), 'object_no': 2, 'car_color': '白色', 'color_conf': 0.8927687406539917}]

end
