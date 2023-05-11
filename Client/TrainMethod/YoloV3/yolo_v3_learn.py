import numpy as np
import tensorflow as tf
from DataSetProcess.config import *
from tensorflow.keras import layers
from keras.callbacks import Callback
from keras.models import load_model
import cv2
import os
# 定义模型
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
class_num=4
class CustomCallback(Callback):
    
    def __init__(self,gui):
        super().__init__()
        self.gui = gui
        self.logs = []
        self.loss=[]
        self.val_loss=[]
        self.acc=[]
        self.val_acc=[]
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # 将字典转换为字符串，并将字符串添加到logs列表中
            logs_str = ', '.join(['{}: {:.4f}'.format(key, value) for key, value in logs.items()])
            # self.logs.append(logs_str)
            # train_info_str='\n'.join(self.logs)
            cv2.waitKey(1)
            self.gui.text_display.append(logs_str)
            self.gui.text_display.ensureCursorVisible()  # 强制更新文本显示
            scroll_bar = self.gui.text_display.verticalScrollBar()
            scroll_bar.setValue(scroll_bar.maximum())
            print(logs)
            #记录损失值和准确率,用于绘图
            self.loss.append(logs['loss'])
            self.val_loss.append(logs['val_loss'])
            self.acc.append(logs['accuracy'])
            self.val_acc.append(logs['val_accuracy'])
            #self.gui.fig.clear()
            #创建图表 fig
            # self.fig=Figure(figsize=(12,6),tight_layout=True)
            # self.fig.suptitle('Training Metrics')
            # if hasattr(self, 'ax1'):
            #     print("delete")
            #     del self.ax1
            # if hasattr(self, 'ax2'):
            #     del self.ax2

            #在Figuire对象中添加两个子图
            
            # self.gui.ax1 = self.fig.add_subplot(1, 2, 1)
            # self.gui.ax1.set_title('Loss')
            # self.gui.ax1.set_xlabel('Epochs')
            # self.gui.ax1.set_ylabel('Loss')
            epochs = range(1, len(self.loss)+1)
            # print("epo:",epochs)
            self.gui.fig.clear()
            # self.gui.ax1 = self.fig.add_subplot(1, 2, 1)
            self.gui.ax1 = self.gui.fig.add_subplot(1, 2, 1)
            self.gui.ax1.set_title('Loss')
            self.gui.ax1.set_xlabel('Epochs')
            self.gui.ax1.set_ylabel('Loss')

            self.gui.ax2 = self.gui.fig.add_subplot(1, 2, 2)
            self.gui.ax2.set_title('Accuracy')
            self.gui.ax2.set_xlabel('Epochs')
            self.gui.ax2.set_ylabel('Accuracy')
            self.gui.ax1.plot(epochs, self.loss, label='Training Loss')
            self.gui.ax1.plot(epochs, self.val_loss, label='Validation Loss')
            self.gui.ax1.legend()
          
            self.gui.ax2.plot(epochs, self.acc, label='Training Accuracy')
            self.gui.ax2.plot(epochs, self.val_acc, label='Validation Accuracy')
            self.gui.ax2.legend()
            self.gui.canvas.draw()
            # cv2.waitKey(1)
            #self.gui.canvas = FigureCanvas(self.gui.fig)
            
            # existing_layout = self.gui.media_display.layout()
            # if existing_layout:
            #     while existing_layout.count():
            #         child = existing_layout.takeAt(0)
            #         if child.widget():
            #             child.widget().deleteLater()

            # # 删除布局管理器
            # del existing_layout
            #cv2.waitKey(1000)
           
            # self.toolbar=NavifationToolbar(self.canvas,self.gui.media_display)
            # layout.addWidget(self.toolbar)
            # self.gui.media_display.setLayout(layout)
            # for i in range(1000):
            #     print("ffsf")
            # plt.clf()
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            # fig.suptitle('Training Metrics')
            # epochs = range(1, len(self.loss)+1)
            # ax1.plot(epochs, self.loss, label='Training Loss')
            # ax1.plot(epochs, self.val_loss, label='Validation Loss')
            # ax1.set_title('Loss')
            # ax1.set_xlabel('Epochs')
            # ax1.set_ylabel('Loss')
            # ax1.legend()
            # ax2.plot(epochs, self.acc, label='Training Accuracy')
            # ax2.plot(epochs, self.val_acc, label='Validation Accuracy')
            # ax2.set_title('Accuracy')
            # ax2.set_xlabel('Epochs')
            # ax2.set_ylabel('Accuracy')
            # ax2.legend()
            # plt.draw()
            # plt.pause(0.01)

# 创建自定义回调函数
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
classes = ["雷军","黄玮龙","马化腾","吴彦祖"]


def train(self,faces,targets):
    # 定义输入张量
    inputs = tf.keras.layers.Input(shape=(416, 416, 3))#三通道的416*416的模型
    # 定义模型输出
    yolo_output = yolo_v3(inputs) #根据输入自动计算输出的类型
    #print(yolo_output.shape)
    model = tf.keras.models.Model(inputs=inputs, outputs=yolo_output)
   # model.summary()
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    train_sets=[]
    label_sets=[]
    faces,targets=ExtractLabels(faces,targets)
    faces = np.array(faces)
    targets=np.array(targets)
    faces = faces.astype('float32') / 255 
    for img,label in zip(faces,targets):
        
        train_sets.append(img)
        temp=preprocess_data(img,label,13,classes)
        label_sets.append(temp)
    print("train_sets.shape: ",train_sets[0].shape)
    print("input.shape: ",inputs.shape)
    print("label_sets.shape: ",label_sets[0].shape)
    print("output.shape",yolo_output.shape)
    print("train.len:",len(train_sets))
    print("label_sets.len",len(label_sets))
    train_sets=np.asarray(train_sets)
    label_sets=np.asarray(label_sets)
    custom_callback = CustomCallback(gui=self)
    model.fit(train_sets,label_sets,batch_size=15, epochs=20,validation_split=0.2,callbacks=[custom_callback])
    model.save_weights("yolov3model.h5")
    self.text_display.append("选择保存权重数据的路径")
    file_path,_=QFileDialog.getSaveFileName(
        None,"保存文件","","Weight Flies(*.h5)"
    )
    if file_path!="":
        model.save(file_path)
    else:
        self.text_display.append("您没有保存权重文件")
# def predict():
#     inputs = tf.keras.layers.Input(shape=(416, 416, 3))#三通道的416*416的模型
#     # 定义模型输出
#     yolo_output = yolo_v3(inputs) #根据输入自动计算输出的类型
#     #print(yolo_output.shape)
#     model = tf.keras.models.Model(inputs=inputs, outputs=yolo_output)
#    # model.summary()
#     model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#     model.load_weights('yolov3model.h5')
#     face=None
#     path=None
#     path=fileChoose()
#     if path =="": path=0
#     print("path=",type(path))
#     cap=cv2.VideoCapture(path)
#     while True:
#         flag,frame=cap.read()#读出视频帧
#         if not flag:#读不到了（播放完了）
#             continue
#         img = frame
#         img=resize_image(img,(416,416))
#         test=[]
#         test.append(img)
#         test=np.asarray(test)
#         result=model.predict(test)
#         print(result.shape)
#         #print(test.shape)
#         print(result[0])
#         # cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
#         # #显示人名，putext（）函数仅支持ASCII
#         # img_pil=Image.fromarray(frame)
#         # draw=ImageDraw.Draw(img_pil)
#         # color=(0,0,255)
#         # org=(x,y-10)
#         # draw.text(org,label,font=font,fill=color)
#         # frame=np.array(img_pil)
#         # #cv2.putText(frame,label,org,cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
#         # cv2.imshow('face',frame)
#         #print(frame.shape)
#         # vw.write(frame)
#         key=cv2.waitKey(1)
#         if key==ord('q'):
#             break
#     cv2.destroyAllWindows()
#     cap.release()
def predict(self):
    #
    self.text_display.append("请选择合适的权重文件")

    #net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")#加载网络
# 加载 COCO 数据集标签
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
        print(classes)
    try:
        path,_= QFileDialog.getOpenFileName(None,"打开文件","","Weight Files(*.weights)")
        #print(path)
        net = cv2.dnn.readNet(path, "yolov3.cfg")#加载网络
    except Exception as ex:
        self.text_display.append(str(ex))
        scroll_bar=self.text_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        return 
    self.text_display.append("请选择进行预测的源视频文件,不选择则打开摄像头")
    try:
        path=None
        path,_= QFileDialog.getOpenFileName(None,"打开视频","","Vedio Files(*.mp4)")
        if path =="": path=0
        cap=cv2.VideoCapture(path)
    except Exception as ex:
        self.text_display.append(str(ex))
        return 
    # 随机生成颜色，用于标记检测到的目标框
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    while True:
        # 从视频流中读取一帧图像
        ret, frame = cap.read()
        
        # 转换图像为blob格式
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(416, 416), swapRB=True, crop=False)
        print(blob.shape)
        net.setInput(blob)
        LayersNames=net.getLayerNames()
        # print(LayersNames)
        # 获取YOLO输出层的名称
        a=net.getUnconnectedOutLayers()
        # print("三个输出层的索引号：\n",a)
        # break
        output_layers = [LayersNames[i- 1] for i in net.getUnconnectedOutLayers()]
        # print("三个输出层名称",output_layers)
        
        # 将blob输入到YOLO网络中，获取目标框和类别信息
        outputs = net.forward(output_layers)
        # print(outputs[0].shape)
        # print(len(outputs))
        # print(outputs)
        # 解析YOLO输出，提取目标框信息和类别概率
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    #print("confidence:",confidence)
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # 应用NMS算法，移除重叠的目标框
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
       # print("indices",indices)
        # 绘制检测到的目标框和类别名称
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness=2)
                cv2.putText(frame, classes[class_ids[i]], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
        
        # 显示结果
        # cv2.imshow('YOLOv3 Real-time Detection', frame)
        self.ShowCVImage(frame)
        cv2.waitKey(1)
        # 按下q键退出
        if self.end:
            self.end=False
            break

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()


# 将模型定义为Keras模型，并打印摘要
# model = tf.keras.models.Model(inputs=inputs, outputs=yolo_output)
# model.summary()
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(faces, target, batch_size=batch_size, epochs=epochs)

# 上述代码中，我们定义了一个YOLOv3的模型，该模型包括了特征提取网络（DarkNet-53）和3个YOLO头。我们首先定义了一个`_conv2d`函数，
# 它包括一个卷积层、一个批量归一化层和一个LeakyReLU激活函数，用于构建卷积层。然后我们定义了一个`_residual_block`函数，
# 它包括2个卷积层和一个残差连接，用于构建残差块。接着我们按照YOLOv3的结构定义了特征提取网络，并使用该网络来提取输入张量的特征。
# 最后我们定义了分类和回归头，以及3个YOLO头，用于检测不同大小的物体。
# 在模型定义完成后，我们可以将其定义为Keras模型，并使用`model.summary()`函数打印模型的摘要信息，以便检查模型结构是否正确。
def preprocess_data(image, targets, grid_size, classes):#
    """
    对输入的图像和目标位置信息进行预处理，将其转换成标签形式的数据
    image: 输入的图像, 为numpy数组格式, 形状为(H, W, C), 其中C表示channel数量
    targets: 目标位置信息，为列表格式，其中每个元素为[x_min, y_min, x_max, y_max, class_id]
             表示一个目标的边界框左上角和右下角的坐标以及物体类别id
    grid_size: 网络划分的网格大小
    classes: 物体类别列表
    """
    height, width, _ = image.shape 
    target_size = (grid_size, grid_size)

    # 生成特征图，并初始化每个网格的标签为零
    label = np.zeros(shape=(grid_size, grid_size, 3, 5+len(classes)), dtype=np.float32)

    # 遍历每个目标位置信息
    for target in targets:
        x_min, y_min, x_max, y_max, class_id = target

        # 计算物体中心点坐标和边界框的宽和高
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        box_width = x_max - x_min
        box_height = y_max - y_min

        # 将目标位置信息转换成特征图上的坐标
        x_center = x_center / float(width) * grid_size#处于第几个格子上
        y_center = y_center / float(height) * grid_size
        box_width = box_width / float(width) * grid_size
        box_height = box_height / float(height) * grid_size

        # 计算目标所在的网格
        x_grid = int(x_center)
        y_grid = int(y_center)

        # 在目标网格的每个边界框中选择一个未被使用的边界框，并将其标记为已用
        for box_idx in range(3):
            if label[y_grid, x_grid, box_idx, 0] == 0:
                label[y_grid, x_grid, box_idx, :4] = [x_center, y_center, box_width, box_height]
                label[y_grid, x_grid, box_idx, 4] = 1.0 # 置信度为1
                try:
                    label[y_grid, x_grid, box_idx, 5+class_id] = 1.0 # 对应类别id的概率为1
                except Exception as ex:
                    print(str(ex)) 
                break
    label_reshaped = np.reshape(label, (13, 13, 3*(5+class_num)))
    return label_reshaped
def ExtractLabels(images,labels):#针对yolov3模型处理标签
    faces=[]
    target=[]
    for img,label in zip(images,labels):
        m_list=label.split()
        img_array=np.frombuffer(img,np.uint8)
        img=cv2.imdecode(img_array,cv2.IMREAD_COLOR) #二进制bits流解码成图片数据
        # print("label=",label)
        # print(img.shape)
        #需要将图片缩放成416*416  
        ratio=1280.0/416
        img=resize_image(img,(416,416))
        if len(m_list)==0:
            continue
        temp=[]
        for i in range(0, len(m_list), 5):
            
            x1=int(m_list[i])/ratio
            y1=int(m_list[i+1])/ratio
            x2=int(m_list[i+2])/ratio
            y2=int(m_list[i+3])/ratio
            id=int(m_list[i+4])
            temp.append([x1,y1,x2,y2,id])
        target.append(temp)
        faces.append(img)
            
 #   cv2.destroyAllWindows()
    return faces,target

            
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