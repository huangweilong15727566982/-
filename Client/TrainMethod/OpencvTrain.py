import cv2
import os
from DataSetProcess.config import *
import numpy as np
import matplotlib.font_manager as fm
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
fourcc=cv2.VideoWriter_fourcc(*'MJPG') #fourcc 表示视频数据流格式 
vw=cv2.VideoWriter("E:\\huang\\shazi1.mp4v",fourcc,25,(720,1280))# 25表示帧率,(1280,20)表示分辨率
font_path=r"./Fonts/simkai.ttf"
font_size=30
font=ImageFont.truetype(font_path,font_size)
def load_data(src): #加载数据 src为图片源路径
    listdir=os.listdir(src) #打开文件列表  将该目录下的文件名全部存入列表listdir
    names=[] #保存所有的文件名
#print(listdir)
    for d in listdir:
        names.append(d)
    print(names)
    faces=[] #保存图像数据
    target=[]#保存名字的索引 通过name[target[i]] 获取第i张图片的分类
    for index,dir in enumerate(names):#遍历  dir是单个文件名(不含后缀)
        for i in range(1,31): #对names30张数据枚举  
            img=cv2.imread(src+"\\%s\\%d.jpg"%(dir,i))# 三维图片
            gray=img[:,:,0] #  转化为灰度图像 二维数组 将第三通道抹去
        #统一缩放为64*64 
            gray=cv2.equalizeHist(gray)#图片均衡化处理
            gray=cv2.resize(gray,dsize=(64,64))

            faces.append(gray)
            target.append(index)
            # print(gray.shape)
    faces=np.asarray(faces)
    target=np.asarray(target)
    print(faces)
    return faces,target,names
#faces 保存了所有图片 target 保存了faces索引的名字索引
def split_data(faces, target,validation_split=0.2):
    dataset_size=len(faces)
    test_size=int(dataset_size*validation_split)
    train_size=dataset_size-test_size
    index=np.arange(dataset_size)
    np.random.shuffle(index)#洗牌
    faces=faces[index]
    target=target[index]
    X_train,X_test=faces[:train_size],faces[train_size:]#测试数据和训练数据
    #目标值
    Y_train,Y_test=target[:train_size],target[train_size:]
    return  X_train,X_test,Y_train,Y_test
def take_photo():
    cap=cv2.VideoCapture("E://huang//tenyuan.mp4")
    face_detector=cv2.CascadeClassifier('E://huang//haarcascade_frontalface_alt.xml')
    filename=1
    flag_write=False
    while True:
        flag,frame=cap.read()
        if flag==False:
            break
        gray=cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)
        faces=face_detector.detector.detectMultiScale(gray,minNeighbors=10)
        for x,y,w,h in faces:
            face =gray[y:y+h,x:x+w]
            cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=[0,0, ])
    cap.release()
def dynamic_recognizer_face(face_recognizer,names):
    cap=cv2.VideoCapture(0)
    #E://huang//video_file//test1.mp4
    #人脸检测
    face_detector=cv2.CascadeClassifier(r"./haarcascade_frontalface_alt.xml")
    while True:
        flag,frame=cap.read()#读出视频帧
        # if not flag:#读不到了（播放完了）
        #     continue
        gray = cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)#灰度化
        faces=face_detector.detectMultiScale(gray,minNeighbors=10)#人脸检测
        for x,y,w,h in faces:# 对人脸进行处理
            face=gray[y:y+h,x:x+w]
            face=cv2.resize(face,dsize=(64,64))#缩放成特定大小
            face=cv2.equalizeHist(face)#均衡化处理
            y_,confidence=face_recognizer.predict(face)#人脸辨识  给出预测标签和置信度
            label=names[y_-1]
            #unicode_label = label.encode("utf-8").decode("unicode_escape")
            print('这个人是：%s。置信度是: %0.1f'%(label,confidence))
            cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
            #显示人名，putext（）函数仅支持ASCII
            # font_path=r"./Fonts/simkai.ttf"
            # font_size=30
            # font=ImageFont.truetype(font_path,font_size)
            img_pil=Image.fromarray(frame)
            draw=ImageDraw.Draw(img_pil)
            color=(0,0,255)
            org=(x,y-10)
            draw.text(org,label,font=font,fill=color)
            frame=np.array(img_pil)
            #cv2.putText(frame,label,org,cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        cv2.imshow('face',frame)
        #print(frame.shape)
        # vw.write(frame)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
#选择训练文件进行预测
def predict(self):
    names=["雷军","黄玮龙","马化腾","吴彦祖"]
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    #face_recognizer.read(r'./trainer_1.yml')
    self.text_display.append("请选择合适的权重文件")
    try:
        path,_= QFileDialog.getOpenFileName(None,"打开文件","","Weight Files(*.yml)")
        #print(path)
        face_recognizer.read(path)
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
        gray = cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)#灰度化
        faces=face_detector.detectMultiScale(gray,minNeighbors=10)#人脸检测
        for x,y,w,h in faces:# 对人脸进行处理
            face=gray[y:y+h,x:x+w]
            face=cv2.resize(face,dsize=(64,64))#缩放成特定大小
            face=cv2.equalizeHist(face)#均衡化处理
            y_,confidence=face_recognizer.predict(face)#人脸辨识  给出预测标签和置信度

            label=names[y_-1]
            #unicode_label = label.encode("utf-8").decode("unicode_escape")
            #print('这个人是：%s。置信度是: %0.1f'%(label,confidence))
            cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
            #显示人名，putext（）函数仅支持ASCII
            img_pil=Image.fromarray(frame)
            draw=ImageDraw.Draw(img_pil)
            color=(0,0,255)
            org=(x,y-10)
            draw.text(org,label,font=font,fill=color)
            frame=np.array(img_pil)
            #cv2.putText(frame,label,org,cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        # cv2.imshow('face',frame)
        self.ShowCVImage(frame)
        #print(frame.shape)
        # vw.write(frame)
        key=cv2.waitKey(1)
        if self.end:
            self.end=False
            break
    cv2.destroyAllWindows()
    cap.release()

def opencv_train(self,images,labels):#图像数据和标签数据
    faces=[]
    target=[]
    faces,target=ExtractLabels(images,labels)#提取标签
    dataset_size=len(faces)

    # #定义损失函数
    # loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # #定义优化器
    # optimizer=tf.keras.optimizers.SGD(learning_rate=0.001)

    # for data in images:
    #     img_array=np.frombuffer(data[0],np.uint8)
    #     img=cv2.imdecode(img_array,cv2.IMREAD_COLOR) #二进制bits流解码成图片数据
        
    #     faces.append(img)
    #     target.append(data[1])
    faces=np.asarray(faces)
    target=np.asarray(target)
    #X_train,X_test,Y_train,Y_test=split_data(faces,target,validation_split=0.2) 
    self.text_display.append(f"训练图像数目{len(faces)}")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # print(target.shape)
    # print(target)
    epochs=50
    batch_size=15
    acc=[]
    val_acc=[]
    for p in range(epochs):#迭代
        X_train,X_test,Y_train,Y_test=split_data(faces,target,validation_split=0.5) 
        if len(X_train)>=batch_size:
            face_batches = np.array_split(X_train, len(X_train) // batch_size) #分批次
            target_batches = np.array_split(Y_train, len(Y_train) // batch_size)   
            for i in range(len(face_batches)):
                batch_faces = face_batches[i]
                batch_target = target_batches[i]
                face_recognizer.update(batch_faces, batch_target)
        if len(X_train) % batch_size != 0:
            last_face_batch =X_train[len(X_train)-((len(X_train) % batch_size)):]
            last_target_batch = Y_train[len(X_train)-((len(X_train) % batch_size)):]
           # print(last_target_batch.shape)
            print(last_target_batch)
            face_recognizer.update(last_face_batch, last_target_batch)
        #计算训练集准确率
        right_num=0
        for item,id in zip(X_train,Y_train):
             y_,confidence=face_recognizer.predict(item)
             if y_==id: right_num+=1
        train_accuracy=float(right_num)/len(X_train)
        #计算测试集准确率
        right_num=0
        for item,id in zip(X_test,Y_test):
             y_,confidence=face_recognizer.predict(item)
             if y_==id: right_num+=1
        test_accuracy=float(right_num)/len(X_test)
        print(train_accuracy,test_accuracy)
        self.text_display.append(f"Epoch {p+1}/{epochs}")
        self.text_display.append(f"accuracy:{train_accuracy} val_accracy:{test_accuracy}")
        scroll_bar=self.text_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        acc.append(train_accuracy)
        val_acc.append(test_accuracy)
        self.fig.clear()
        self.ax1=self.fig.add_subplot(1,1,1)#一行一列的第一个子图
        self.ax1.set_title('Accuracy')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Accuracy')
        epoch=[]
        for k in range(p+1):
            epoch.append(k+1)
        print(epoch)
        self.ax1.plot(epoch,acc,label='Training Accracy')
        self.ax1.plot(epoch,val_acc,label='Validation Accuracy')
        cv2.waitKey(1)
        self.ax1.legend()
        self.canvas.draw()
    self.text_display.append("选择保存权重数据的路径")
    file_path,_=QFileDialog.getSaveFileName(
        None,"保存文件","","Weight Flies(*.yml)"
    )
    if file_path!="":
       face_recognizer.write(file_path)#保存训练好的文件
    else:
        self.text_display.append("您没有保存权重文件")
    #face_recognizer.write(r'./trainer_1.yml')#保存训练好的文件

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
            id=int(m_list[i+4])
            roi=img[y1:y2,x1:x2]
            roi=resize_image(roi,(64,64))
            # cv2.imshow("roi",roi)
            # cv2.waitKey(0)
            roi=cv2.cvtColor(roi,code=cv2.COLOR_BGR2GRAY)
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
#if __name__=='__main__':
    #print(cv2.__version__) 
    #1.动态采集人脸
    #take_photo()
    #2.加载数据，返回目标值
    #faces,target,names=load_data("E://huang//pic_file") # 
    #faces,target=DB.getPicture()
    #3.加载算法
    #face_recognizer=cv2.face.EigenFaceRecognizer_create()
    #face_recognizer=cv2.face.FisherFaceRecognizer_create()
    #face_recognizer=cv2.face.createLBPHFaceRecognizer()
    #face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    #4.训练
    #训练需要的数据 图片，以及图片的分类 target数组中存放分类的编号
    #face_recognizer.train(faces,target) 
    #可以将训练结果保存在train.yml中
    #face_recognizer.write('trainer.yml')
    #5. 动态加载数据
    #names=["玮龙","陈波","瑞祥"]
    #dynamic_recognizer_face(face_recognizer,names)
      
 # cv2.putText(frame,text=unicode_label,org=(x,y-10),
            #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.5,color=[0,0,255],thickness=2,fontPr)
            # cv2.putText(frame, text=unicode_label, org=(200, 250),
            # fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5,
            # color=[0, 0, 255], thickness=2, 
            # bottomLeftOrigin=False)
