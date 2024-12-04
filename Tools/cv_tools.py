import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import sys
import os


"""
功能:
    调试图像的hsv以提取想要的颜色
参数
    img_BGR:        opencv格式的图像格式
    distribution:   图像放置方式
    func:           滑动条触发函数
    size:           显示的图像的尺寸
    part:           滑动条是否与图像分离            
返回值
"""
def empty(a):
    pass
def Get_Appoint_Color(img_BGR,distribution=1,func=empty,size=[200,200],part=0):
    if not isinstance(img_BGR,np.ndarray):
        print("这不是有效的图像")
        return -1
    print("点击'q'退出")
    img_BGR = cv2.resize(img_BGR,size)
    imgHSV=cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV)
    cv2.namedWindow("TrackBars")                               # 生成一个叫 name 的窗口
    if not part:
        cv2.resizeWindow("TrackBars",[size[0]*4,size[1]*2])          # 改变其窗口大小
    else:
        cv2.resizeWindow("TrackBars",[640,240]) 
    cv2.createTrackbar("Hue Min","TrackBars",0,179,func)       # 为窗口创建一个0~179的滑动条,0为起始位置
    cv2.createTrackbar("Hue Max","TrackBars",179,179,func)     # 为窗口创建一个0~179的滑动条
    cv2.createTrackbar("Sat Min","TrackBars",0,255,func)      # 为窗口创建一个0~255的滑动条
    cv2.createTrackbar("Sat Max","TrackBars",255,255,func)     # 为窗口创建一个0~255的滑动条
    cv2.createTrackbar("val Min","TrackBars",0,255,func)      # 为窗口创建一个0~255的滑动条
    cv2.createTrackbar("val Max","TrackBars",255,255,func)     # 为窗口创建一个0~255的滑动条
    while True:
        h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("val Max", "TrackBars")


        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        mask  = cv2.inRange(imgHSV,lower,upper)                # 相当与蒙版，用来与源图片比较
        imgResult = cv2.bitwise_and(img_BGR,img_BGR,mask=mask)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if distribution:
            horizontal_image_1 = np.hstack((img_BGR,imgHSV))
            horizontal_image_2 = np.hstack((mask,imgResult))
            vertical_image = np.hstack((horizontal_image_1, horizontal_image_2))
        else:
            horizontal_image_1 = np.hstack((img_BGR,imgHSV))
            horizontal_image_2 = np.hstack((mask,imgResult))
            vertical_image = np.vstack((horizontal_image_1, horizontal_image_2))
        if part==1:
            cv2.imshow("src-hsv/mask/result",vertical_image)
        else:
            cv2.imshow('TrackBars',vertical_image)
        key = cv2.waitKey(1)                                    # 10表示暂停 10ms ,0表示一直停止
        if key & 0xFF== ord('q'):
            cv2.destroyAllWindows()
            print("已退出")
            break
'''
功能:
    显示多个图像在一个窗口
参数:
    imgs:列表,包含你要显示的图像序列
    window_name:你要显示窗口的名称
    lines:每行显示的图像数
    scale:图像的比例
'''
def CV2_Imgs(imgs,window_name="imgs",lines=3,scale=1):
    print("点击'q'退出")
    max_height = int(max(image.shape[0] for image in imgs)*scale)
    max_width = int(max(image.shape[1] for image in imgs)*scale)
    resized_images = []
    for image in imgs:
        if len(image.shape) == 2:  # 如果是灰度图像，扩展为三通道
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resized_image = cv2.resize(image, (max_width, max_height))
        resized_images.append(resized_image)
    num_rows = (len(resized_images) + lines - 1) // lines

    grid_rows = []
    for i in range(num_rows):
        start_idx = i * lines
        end_idx = min(start_idx + lines, len(resized_images))
        row_images = resized_images[start_idx:end_idx]
        if len(row_images) < lines:
            # 如果最后一行图像数量不足，用空白图像补齐
            for _ in range(lines - len(row_images)):
                row_images.append(np.zeros_like(resized_images[0]))
        grid_rows.append(np.hstack(row_images))
    
    combined_image = np.vstack(grid_rows)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined_image)
    key = cv2.waitKey(0)                                    # 10表示暂停 10ms ,0表示一直停止
    if key & 0xFF== ord('q'):
        cv2.destroyAllWindows()
        print("已退出")
"""
功能:
    在图像上添加中文文本。
参数:
    image: 输入图像 (OpenCV格式)
    text: 要添加的文本
    position: 文本位置 (x, y)
    font_path: 字体文件路径
    font_size: 字体大小
    color: 文本颜色 (默认白色)
返回值: 
    添加文本后的图像
"""
def TTF_Abs_Path():
    abs_path = os.path.abspath(__file__)
    abs_path = abs_path[0:-11] 
    abs_path += "files/SIMHEI.TTF"
    return abs_path
def Print_Chinese(image, text, position, font_size, color=(255, 255, 255), font_path = TTF_Abs_Path()):
    # 将OpenCV图像转换为PIL图像
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 创建绘图对象
    draw = ImageDraw.Draw(pil_img)
    # 加载字体
    font = ImageFont.truetype(font_path, font_size)
    # 在图像上绘制文本
    draw.text(position, text, font=font, fill=color)
    # 将PIL图像转换回OpenCV格式
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv_img


if __name__ == '__main__':
    img = np.random.randint(0, 255, size=(300,300,3), dtype=np.uint8)
    img1 = Print_Chinese(img,"参asdf撒反对123213数",(0,0),25,(0,0,255))
    CV2_Imgs([img1])
    # CV2_Imgs([img,img,img,img],window_name="imgs",lines=3,scale=0.9)
    Get_Appoint_Color(img,distribution=0,size=[300,300],part=1)
    
    
    
    