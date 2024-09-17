import cv2
import keyboard
import sys
import os
import numpy as np
import socket
import cv2
import struct
import time
import msvcrt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
sys.path.append(r'./DataSetProcess') 
import DRAWER as dw
from config import *
#from DRAWER import * 
# sys.path.append(r'/DataSetProcess') 
def importTask(self):
    path=fileChoose()
    print(path)
    if path=="":
        path=0
    cap=cv2.VideoCapture(path)
    # tasksets=[]#task的数组
    taskset=[]#存放图片的二进制bit流集合的数组
    tasknum=0
    new_size = (832,832)
    # task_id=0
    # cv2.namedWindow("Virtual Window")
    # cv2.moveWindow("Virtual Window", 0, 0)
    # cv2.setWindowProperty("Virtual Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.setWindowProperty("Virtual Window", cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Virtual Window", 0, 0)
    while True:
        flag,frame=cap.read()
        if flag==False:
            break 
        frame = resize_image(frame, new_size)#处理成 832*832
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        p=cv2qt(self,img)
        self.media_display.setPixmap(QPixmap.fromImage(p))
        key=cv2.waitKey(22)    
        if self.capture:
            tasknum+=1
            retval,buffer=cv2.imencode('.jpg',frame)
            binary_data=buffer.tobytes()
            taskset.append(binary_data)
            self.text_display.append(f"已收集{tasknum}个图片")
            scroll_bar = self.text_display.verticalScrollBar()
            scroll_bar.setValue(scroll_bar.maximum())
            self.capture=False
        elif self.end :break  
    return taskset

#数据标注  
#许多二进制数据
# 打开图片画框框 存入数组中 
def DataAnnotation(self,task,table):#数据标注  许多二进制数据 
    print("任务数量：",len(task))
    self.text_display.append(f"任务数量：{len(task)}")
    self.text_display.append("按q结束标注,按z撤销上一次标注")
    for binary_data,table_name in zip(task,table):
        self.text_display.append(f"请标注出目标：{table_name}")
        scroll_bar = self.text_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
       # notice(table_name)#弹窗通知
        img_array=np.frombuffer(binary_data,np.uint8)
        img1=cv2.imdecode(img_array,cv2.IMREAD_COLOR)
        img2=img1.copy()
        img3=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        img3=resize_image(img3,(832,832))
        p=cv2qt(self,img3)
        self.media_display.setPixmap(QPixmap.fromImage(p))
        rect_drawer = dw.RectangleDrawer(img2)
        rec,label=rect_drawer.run()#返回的是矩形框数据
        operator="CommitDataSet"#操作码 提交一条数据  命令长度+命令+图片长度+图片+string
        operator=bytes(operator,'utf-8')
        message=struct.pack('>I', len(operator)) + operator
        message+=struct.pack('>I',len(binary_data))+binary_data
        table_name=bytes(table_name,'utf-8')
        message+=struct.pack('>I',len(table_name))+table_name
        string=""
        for item1,item2 in zip(rec,label): #字符串格式 "位置+分类号"
            string+=str(item1[0][0])
            string+=" "
            string+=str(item1[0][1])
            string+=" "
            string+=str(item1[1][0])
            string+=" "
            string+=str(item1[1][1])
            string+=" "
            string+=str(item2)
            string+=" "
        string=bytes(string,'utf-8')
        message+=struct.pack('>I',len(string))+string
        message+=struct.pack('>I',0xffffffff)
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.connect((self.HOST,self.PORT))#建立连接
            s.sendall(message)#发送数据给服务端
            s.recv(1)#接受确认
            s.close()#客户端主动断开连接
        #创建连接 输送数据给服务端并加入数据集中
#制作成消息 发送给服务端
def cv2qt(main_window,img):
    h,w,ch=img.shape
#将Opencv图像转为Qt格式
    bytes_per_line=ch*w
    convert_to_Qt_format = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(main_window.media_display.width(), main_window.media_display.height(), Qt.KeepAspectRatio)
    return p
