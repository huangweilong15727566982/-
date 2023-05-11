import cv2
import os
import numpy as np
def load_data():
    listdir=os.listdir("E:\\huang\\pic_file")
    names=[]
#print(listdir)
    for d in listdir:
        names.append(d)
    print(names)
    faces=[]
    target=[]
    for index,dir in enumerate(names):
        for i in range(1,31):
            img=cv2.imread("E:\\huang\\pic_file\\%s\\%d.jpg"%(dir,i))# 三维图片
            gray=img[:,:,0] #二维数组
        #统一缩放为64*64 
            gray=cv2.resize(gray,dsize=(64,64))
            faces.append(gray)
            target.append(index)
            print(gray.shape)
    faces=np.asarray(faces)
    target=np.asarray(target)
    print(faces)
    return faces,target,names
#faces 保存了所有图片 target 保存了faces索引的名字索引
def split_data(faces, target):
    index=np.arange(210)
    np.random.shuffle(index)#洗牌
    faces=faces[index]
    target=target[index]
    X_train,X_test=faces[:150],faces[150:]#测试数据和训练数据
    #目标值
    Y_train,Y_test=target[:150],target[150:]
    return  X_train,X_test,Y_train,Y_test

if __name__=='__main__':
    #1.加载数据，返回目标值
    faces,target,names=load_data()
    #2.数据拆分
    X_train,X_test, Y_train,Y_test = split_data(faces, target)
    #3.加载算法
    face_reongnizer=cv2.face.EigenFaceRecognizer_create()
    #4.训练
    face_reongnizer.train(X_train,Y_train)
    #5.使用算法进行预测
    for face in X_test:
        #返回预测值是数字
        #返回两个,第一个是类别，第二个是执行度(距离)，越低越好!
        y_,confidence=face_reongnizer.predict(face)
        name=names[y_]
        print("这个人是：",name)
        cv2.imshow('face',face)
        key=cv2.waitKey(0)

