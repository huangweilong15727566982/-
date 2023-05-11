import numpy as np
import tensorflow as tf
# from .DataSetProcess.config import *
# from ...DataSetProcess.config import*
from tensorflow.keras import layers
from keras.models import load_model
import cv2
import os
colors = np.random.uniform(0, 255, size=(2, 3))
def resize_image(image, new_size):#填充右下角
    # 获取原始图像的宽度和高度
    h, w = image.shape[:2]
    #print(h,w)
    # 计算缩放比例
    ratio = min(float(new_size[0])/w, float(new_size[1])/h)
    # 缩放图像
    resized_image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    # 创建填充后的画布
    canvas = 255 * np.ones((new_size[1], new_size[0], 3), dtype=np.uint8)
    h_, w_ = resized_image.shape[:2]
    # 将图像置于填充画布左上角
    x = 0
    y = 0
    canvas[y:y+h_, x:x+w_, :] = resized_image[:, :, :]
    return canvas
def predict():
    inputs = tf.keras.layers.Input(shape=(416, 416, 3))#三通道的416*416的模型
    # 定义模型输出
    yolo_output = yolo_v3(inputs) #根据输入自动计算输出的类型
    #print(yolo_output.shape)
    model = tf.keras.models.Model(inputs=inputs, outputs=yolo_output)
   # model.summary()
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.load_weights('yolov3model.h5')
    face=None
    path=None
    # path=fileChoose()
    # if path =="": path=0
    cap=cv2.VideoCapture(r"E:/huang/video_file/wuyanzu.mp4")
    # img=cv2.imread(r"E:/huang/wu.jpg")
    # img=resize_image(img,(416,416))
    while True:
        ret,img=cap.read()
        img=resize_image(img,(416,416))
        test=[]
        test.append(img)
        test=np.asarray(test)
        result=model.predict(test)
        result=result[0]
        result=result.reshape(507,7)
        print(result.shape)
        boxes,confidences,class_ids=decode(result,img)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.1, nms_threshold=0.1)
        print("indices",len(indices))
        print(indices)
        # 绘制检测到的目标框和类别名称
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=2)
                cv2.putText(img, classes[class_ids[i]], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
        
        # 显示结果
        cv2.imshow('YOLOv3 Real-time Detection', img)
    # 按下q键退出
        if cv2.waitKey(1) == ord('q'):
            break

# 清理资源

    cv2.destroyAllWindows()
def decode(outputs,frame):
    boxes = []
    confidences = []
    class_ids = []
    for detection in outputs:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence >= 0.5:
            print("confidence:",confidence)
            center_x = int(detection[0] * frame.shape[1])
            center_y = int(detection[1] * frame.shape[0])
            width = int(detection[2] * frame.shape[1])
            height = int(detection[3] * frame.shape[0])
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    print("box",len(boxes))
    print(boxes)
    return boxes,confidences,class_ids
   
# def getpos(grid_x,grid_y,item):
#     grid_size=13
#     input_width=416
#     input_height=416
#     x = (grid_x + sigmoid(item[0])) / 13
#     y = (grid_y + sigmoid(item[1])) / grid_size
#     width = anchor_w * np.exp(output[..., 2]) / input_width
#     height = anchor_h * np.exp(output[..., 3]) / input_height
#     x1 = (x - width / 2) * input_width
#     y1 = (y - height / 2) * input_height
#     x2 = (x + width / 2) * input_width
#     y2 = (y + height / 2) * input_height

def yolo_v3(inputs):#输入为 416*416*3的图片数据
    # 定义卷积层
    def _conv2d(inputs, filters, kernel_size, strides, padding="SAME"):
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        return x
    #该函数用于定义卷积层，
    # 其中`inputs`表示输入张量，
    # `filters`表示卷积核数量，
    # `kernel_size`表示卷积核大小，`strides`表示卷积步幅，
    # `padding`表示填充方式。
    # 函数返回一个卷积、归一化和激活之后的张量。 特征明显

    # 定义残差块（Residual Block）
    def _residual_block(inputs, filters):
        shortcut = inputs
        x = _conv2d(inputs, filters // 2, 1, 1)
        x = _conv2d(x, filters, 3, 1)
        x = x + shortcut
        return x
# 该函数用于定义残差块，其中`inputs`表示输入张量，`filters`表示卷积核数量。
# 函数先将输入张量保存在`shortcut`变量中，然后对输入张量进行2次卷积操作，
# 每次卷积后都进行一次卷积、归一化和激活操作，最后将结果和`shortcut`张量相加得到输出张量。
    # 定义特征提取网络（DarkNet-53）
    x = _conv2d(inputs, 32, 3, 1)
    x = _conv2d(x, 64, 3, 2)
    x = _residual_block(x, 64)
    x = _conv2d(x, 128, 3, 2)
    for i in range(2):
        x = _residual_block(x, 128)
    x = _conv2d(x, 256, 3, 2)
    for i in range(8):
        x = _residual_block(x, 256)
    route1 = x
    x = _conv2d(x, 512, 3, 2)
    for i in range(8):
        x = _residual_block(x, 512)
    route2 = x
    x = _conv2d(x, 1024, 3, 2)
    for i in range(4):
        x = _residual_block(x, 1024)
    route3 = x

    # 定义分类和回归头
    x = _conv2d(route3, 512, 1, 1)
    x = _conv2d(x, 1024, 3, 1)
    x = _conv2d(x, 512, 1, 1)
    x = _conv2d(x, 1024, 3, 1)
    x = _conv2d(x, 512, 1, 1)

    # 定义yolo头
    yolo_head = _conv2d(x, 1024, 3, 1)
    yolo_output = tf.keras.layers.Conv2D(3 * (5 + len(classes)), 1, strides=1, padding="SAME", use_bias=True)(yolo_head)

    #这段代码是在创建一个 YOLO （You Only Look Once）模型的头部，YOLO 是一种常用的目标检测算法。
    # 这个头部由两个部分组成，第一个部分是在输入张量 x 上使用 3x3 大小的卷积核和 1024 个卷积核进行卷积的结果。
    # 第二个部分是在第一个部分卷积后得到的张量上使用 1x1 大小的卷积核进行卷积，并输出一个大小为 3 * (5+len(classes)) 的张量。
    # 其中5是由 YOLO 模型中每个检测框预测的坐标信息（中心点坐标、宽度和高度）加上每个检测框分类信息所组成的（通常使用概率向量）。
    # len(classes) 是指分类器所检测的物体类别数量，这里将其加入了每个检测框的预测中。
    # 换句话说，这个头部输出的是一张图片中所有可能的检测框的预测，其中每个预测包含检测框的坐标信息和分类信息。

    return yolo_output
classes = ["target","unkonw"]
predict()
