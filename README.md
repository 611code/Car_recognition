# 车辆识别系统

原作者github链接[611code/Chinese_license_plate_detection_recognition: yolov5 车牌检测 车牌识别 中文车牌识别 检测 支持12种中文车牌 支持双层车牌](https://github.com/611code/Chinese_license_plate_detection_recognition)

这里fork是因为感觉代码写的很好，但作为小白学习难度太大，此处记录如何看懂代码

文件为

My_Learn_Detect_Code.ipynb

My_Learn_Detect_Note.md

**目前支持车辆检测+车牌检测识别**

环境要求: python >=3.6  pytorch >=1.7

#### **图片测试demo:**

```
python Car_recognition.py --detect_model weights/detect.pt  --rec_model weights/plate_rec_color.pth --image_path imgs --output result
```

测试文件夹imgs，结果保存再 result 文件夹中

![Image text](image/test.jpg)

## **检测训练**

1. **下载数据集：**  获取数据集+V：we0091234 （注明来意）**收费30 介意勿扰**
   数据集格式为yolo格式：

   ```
   label x y w h  pt1x pt1y pt2x pt2y pt3x pt3y pt4x pt4y
   ```

   关键点依次是（左上，右上，右下，左下）
   坐标都是经过归一化，x,y是中心点除以图片宽高，w,h是框的宽高除以图片宽高，ptx，pty是关键点坐标除以宽高

   车辆标注不需要关键点 关键点全部置为-1即可
2. **修改 data/widerface.yaml    train和val路径,换成你的数据路径**

   ```
   train: /your/train/path #修改成你的路径
   val: /your/val/path     #修改成你的路径
   # number of classes
   nc: 3                #这里用的是3分类，0 单层车牌 1 双层车牌 2 车辆

   # class names
   names: [ 'single_plate','double_plate','Car'] 

   ```
3. **训练**

   ```
   python3 train.py --data data/plateAndCar.yaml --cfg models/yolov5n-0.5.yaml --weights weights/detect.pt --epoch 250
   ```

   结果存在run文件夹中

## **车牌识别训练**

车牌识别训练链接如下：

[车牌识别训练](https://github.com/we0091234/crnn_plate_recognition)

## References

* [https://github.com/we0091234/Chinese_license_plate_detection_recognition](https://github.com/we0091234/Chinese_license_plate_detection_recognition)
* [https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)
* [https://github.com/meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch)

## TODO

车型，车辆颜色，品牌等。

## 联系

**有问题可以提issues 或者加qq群：769809695（三群新开） 823419837(已满) 837982567（已满）  询问**
