import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model
from keras.callbacks import Callback
from DataSetProcess.config import *
import matplotlib.font_manager as fm
from PIL import ImageFont, ImageDraw, Image
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout,QDialog,QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavifationToolbar
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import copy
font_path=r"./Fonts/simkai.ttf"
font_size=30
font=ImageFont.truetype(font_path,font_size)
#from object_detection.utils.visualization_utils import draw_bounding_boxes_on_image
# 设置超参数
batch_size = 50#表示每次训练中使用的样本数量
epochs = 20   #迭代次数
num_classes =4#表示分类的类别数目。
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

def train(self,images,labels):
    custom_callback = CustomCallback(gui=self)
    faces,target=ExtractLabels(images,labels)
    # for data in DataSet:
    #     img_array=np.frombuffer(data[0],np.uint8)
    #     img=cv2.imdecode(img_array,cv2.IMREAD_GRAYSCALE) #二进制bits流解码成图片数据
    #     faces.append(img)
    #     target.append(data[1])
    faces=np.asarray(faces)
    target=np.asarray(target) 
    
    # 对图像进行归一化
    faces = faces.astype('float32') / 255 
    #标签转换成独热编码
    #print(target)
    target=tf.keras.utils.to_categorical(target, num_classes)#target 只有 0和1 二分类
    #定义卷积神经网络模型
    model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),#输入层为彩色3通道图像 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),#输出成
    layers.Dense(num_classes, activation='softmax')
    ])
    # for i in range(100):
    #     cv2.waitKey(1000)
    #     self.text_display.append("111")

    # 编译模型
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    #训练模型
    history=model.fit(faces, target, batch_size=batch_size, epochs=epochs, validation_split=0.2,verbose=2,callbacks=[custom_callback])#callbacks=[custom_callback]
    self.text_display.append("选择保存权重数据的路径")
    file_path,_=QFileDialog.getSaveFileName(
        None,"保存文件","","Weight Flies(*.h5)"
    )
    if file_path!="":
        model.save(file_path)
    else:
        self.text_display.append("您没有保存权重文件")
# def predict():
#     model =load_model('model1.h5')

#     # 定义滑动窗口的大小和步长
#     window_size = (128, 128)
#     stride = 64
#     # 遍历图像块并进行预测
#     cap=cv2.VideoCapture(0)
#     while True:
#         flag,frame=cap.read()#读出视频帧
#         if not flag:#读不到了（播放完了）
#             continue
#         # for x,y,w,h in faces:# 对人脸进行处理
#         #     face=gray[y:y+h,x:x+w]
#         #     face=cv2.resize(face,dsize=(64,64))#缩放成特定大小
#         #     face=cv2.equalizeHist(face)#均衡化处理
#         #     y_,confidence=face_recognizer.predict(face)#人脸辨识  给出预测标签和置信度
#             #unicode_label = label.encode("utf-8").decode("unicode_escape")
#         id,pos=f(frame,window_size,stride,model)
#         names=["玮龙","陈波","瑞祥"]
#         for x,y in zip(id,pos):#绘制矩形框
#             cv2.rectangle(frame,pt1=(y[0],y[1]),pt2=(y[0]+window_size[0],y[1]+window_size[0]),color=[0,0,255],thickness=2)
#         # #显示人名，putext（）函数仅支持ASCII
#             font_path=r"./Fonts/simkai.ttf"
#             font_size=30
#             font=ImageFont.truetype(font_path,font_size)
#             img_pil=Image.fromarray(frame)
#             draw=ImageDraw.Draw(img_pil)
#             color=(0,0,255)
#             org=(y[0],y[1]-10)
#             draw.text(org,names[x-1],font=font,fill=color)
#             frame=np.array(img_pil)
#         cv2.imshow('face',frame)
#         # #print(frame.shape)
#         # # vw.write(frame)
#         key=cv2.waitKey(1)
#         if key==ord('q'):
#             break
#     cv2.destroyAllWindows()
#     cap.release()

# def f(frame,window_size,stride,model):#滑动窗口机制
#     label=[]
#     pos=[]
#     img = cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)#灰度化
#     for x in range(0, img.shape[1] - window_size[1], stride):
#         for y in range(0, img.shape[0] - window_size[0], stride):
#             img_block = img[y:y+window_size[0], x:x+window_size[1]]
            
#             # 预处理输入
#             img_block = cv2.resize(img_block, (64, 64))
#             img_block = img_block.astype(np.float32) / 255.0
#             img_block = np.expand_dims(img_block, axis=0)
            
#             # 进行预测
#             predictions = model.predict(img_block)
#             print(predictions)
#             class_id = np.argmax(predictions[0])
#             print(class_id)
#             threshold=0.9
#             if predictions[0][class_id]>threshold: #置信度大于0.5
#                 print(predictions[0][class_id])
#                 label.append(class_id)
#                 pos.append((x,y))
  #  return label,pos
            # 根据阈值筛选结果
            # threshold = 0.5
            # selected_indices = np.where(scores > threshold)[0]
            # selected_boxes = boxes[selected_indices]
            # selected_classes = classes[selected_indices]
            # selected_scores = scores[selected_indices]

            # 在图像上绘制边界框
           # draw_bounding_boxes_on_image(image, selected_boxes, color='red', thickness=2, display_str_list=selected_classes, use_normalized_coordinates=True)

            # 展示图像
            # cv2.imshow('Image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
           # print(pred)
            #ret.append(pred)
    #print(ret)
# predict()
def ExtractLabels(images,labels):
    faces=[]
    target=[]
    for img,label in zip(images,labels):
        m_list=label.split()
        img_array=np.frombuffer(img,np.uint8)
        img=cv2.imdecode(img_array,cv2.IMREAD_COLOR) #二进制bits流解码成图片数据
        print("label=",label)
        for i in range(0, len(m_list), 5):
            x1=int(m_list[i])
            y1=int(m_list[i+1])
            x2=int(m_list[i+2])
            y2=int(m_list[i+3])
            id=int(m_list[i+4])-1
            roi=img[y1:y2,x1:x2]
            roi=resize_image(roi,(64,64))
            # cv2.imshow("roi",roi)
            # cv2.waitKey(0)
            #roi=cv2.cvtColor(roi,code=cv2.COLOR_BGR2GRAY)
            faces.append(roi)
            target.append(id)
 #   cv2.destroyAllWindows()
    return faces,target

            
def resize_image(image, new_size):
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
def predict(self):
    names=["雷军","黄玮龙","马化腾","吴彦祖"]
    self.text_display.append("请选择合适的权重文件")
    try:
        path,_= QFileDialog.getOpenFileName(None,"打开文件","","Weight Files(*.h5)")
        print(path)
        model =load_model(path)
    except Exception as ex:
        self.text_display.append(str(ex))
        scroll_bar=self.text_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        return 
    self.text_display.append("请选择进行预测的源视频文件,不选择则打开摄像头")
    try:
        path=None
        path,_= QFileDialog.getOpenFileName(None,"打开视频","","Vedio Filse(*.mp4)")
        if path =="": path=0
        cap=cv2.VideoCapture(path)
    except Exception as ex:
        self.text_display.append(str(ex))
        return 
    face_detector=cv2.CascadeClassifier(r"./TrainMethod/haarcascade_frontalface_alt.xml")
    while True:
        flag,frame=cap.read()#读出视频帧
        if not flag:#读不到了（播放完了）
            continue
        gray = frame
        gray1=cv2.cvtColor(gray,code=cv2.COLOR_BGR2GRAY)
        faces=face_detector.detectMultiScale(gray1,minNeighbors=10)#人脸检测
        for x,y,w,h in faces:# 对人脸进行处理
            face=gray[y:y+h,x:x+w]
            face=cv2.resize(face,dsize=(64,64))#缩放成特定大小
            #face=cv2.equalizeHist(face)#均衡化处理
            faces = []
            faces.append(face)
            faces=np.asarray(faces)
            faces.astype('float32') / 255 
            prediction=model.predict(faces) #人脸辨识  给出预测标签和置信度
            print(prediction.shape," \n")
            print(prediction,"\n")
            class_id = np.argmax(prediction[0])#求较大置信度的分类下标
            confidence=prediction[0][class_id]
            label=names[class_id]
            print(f"这个人是：{label} ",f"置信度为：{confidence} \n")
            cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
            #显示人名，putext（）函数仅支持ASCII
            img_pil=Image.fromarray(frame)
            draw=ImageDraw.Draw(img_pil)
            color=(0,0,255)
            org=(x,y-10)
            draw.text(org,label,font=font,fill=color)
            frame=np.array(img_pil)
            #cv2.putText(frame,label,org,cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        #cv2.imshow('face',frame)
        self.ShowCVImage(frame)
        key=cv2.waitKey(1)
        # if key==ord('q'):
        #     break
        if self.end: 
            self.end=False
            break
    cv2.destroyAllWindows()
    cap.release()
        #     label=names[y_-1]
        #     #unicode_label = label.encode("utf-8").decode("unicode_escape")
        #     print('这个人是：%s。置信度是: %0.1f'%(label,confidence))
        #     cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
        #     #显示人名，putext（）函数仅支持ASCII
        #     img_pil=Image.fromarray(frame)
        #     draw=ImageDraw.Draw(img_pil)
        #     color=(0,0,255)
        #     org=(x,y-10)
        #     draw.text(org,label,font=font,fill=color)
        #     frame=np.array(img_pil)
        #     #cv2.putText(frame,label,org,cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        # cv2.imshow('face',frame)
        # #print(frame.shape)
        # # vw.write(frame)
        # key=cv2.waitKey(1)
        # if key==ord('q'):
        #     break
    # cv2.destroyAllWindows()
    # cap.release()