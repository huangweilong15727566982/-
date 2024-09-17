#from DataSetProcess import DataSetProcess as DSP
from TrainMethod import OpencvTrain as OT
from TrainMethod import CnnTrain as CT
from TrainMethod import FasterRCNN as FCT
from TrainMethod.YoloV3 import yolo_v3_learn as YL
import sys
from DataSetProcess import DataSetProcess as DSP
import socket
import cv2
import struct
import numpy as np
#导入任务包
#收集数据集
#数据标注
#拉取训练
#和服务器建立连接
HOST='192.168.227.193'#服务器的IP地址
PORT=12345#服务器的端口号
#每需要一个请求时再建立连接
while True: #类似于控制台
    x=int(input("输入操作数："))
    # 1.任务包制作导入
    # 2.任务包提取
    # 3.数据包提交
    # 4.拉取数据包
    if x==1:#任务包制作导入
        #获取表项信息
        operator="GetTableInfo"
        operator=bytes(operator,'utf-8')
        message=struct.pack('>I', len(operator)) + operator
        #
        tables=[]#存储获取的表单信息
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.connect((HOST,PORT))#建立连接
            s.sendall(message)#发送数据给服务端
            print(s.recv(1))
            #接受表信息
            while True:
                Len=struct.unpack('>I',s.recv(4))[0]#接受下一张bit流的长度
                if Len==0xffffffff:break #表示结束位
                string=s.recv(Len)
                string=string.decode('utf-8')
                tables.append(string)
            s.sendall(b'1')#向服务器确认
            s.close()#客户端主动断开连接
        for item in tables:
            print(item,'\n')
        table=input("选择导入的表项,仅从已有的表项中选择,输入正确的表名:\n")
        taskset=DSP.importTask()
        print("实际收集图片：",len(taskset))
        operator="ImportTask"#操作码
        operator=bytes(operator,'utf-8')
        table=bytes(table,'utf-8')
        message=struct.pack('>I', len(operator)) + operator
        message+=struct.pack('>I',len(table))+table
        mess=message
        #削减传输长度 每次最多传输4张图片
        num=0
        end= 0xffffffff
        for img in taskset:
            mess+=struct.pack('>I', len(img)) + img
            num+=1
            if num==4:
                mess +=struct.pack('>I',end)
                num=0
                with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
                    s.connect((HOST,PORT))#建立连接
                    s.sendall(mess)#发送数据给服务端
                    s.recv(1)#确认收到
                    s.close()#客户端主动断开连接
                mess=message
        mess+=struct.pack('>I',end)
        print("主体已传输完毕")
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.connect((HOST,PORT))#建立连接
            s.sendall(mess)#发送数据给服务端
            s.recv(1)#确认收到
            s.close()#客户端主动断开连接

    elif x==2: #数据标注
        #从服务器拉取任务包
        operator="AcquireTask"
        operator=bytes(operator,'utf-8')
        message=struct.pack('>I', len(operator)) + operator
        #print(message.encode('utf-8').decode())
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.connect((HOST,PORT))#建立连接
           # 在使用`s.sendall()`发送字符串数据之前，需要将字符串转换为字节串（bytes）。
           # 因为`s.sendall()`函数只接受字节串数据。可以使用字符串的`encode()`方法将其转换为字节串。
            s.sendall(message)#发送数据给服务端命令
            #接受服务器发来的数据包
            taskset=[]
            tables=[]
            while True:
                Len=struct.unpack('>I',s.recv(4))[0]#接受下一张bit流的长度
                if Len==0xffffffff:break #表示结束位
                img=s.recv(Len)
                taskset.append(img)
                Len=struct.unpack('>I',s.recv(4))[0]#接受table名称的长度
                table=s.recv(Len)
                table=table.decode('utf-8')
                tables.append(table)
            s.sendall(b'1')#确认号
            s.close()#客户端主动断开连接
        if len(taskset)==0:
             DSP.notice("No Task!")#弹窗通知
        DSP.DataAnnotation(taskset,tables)#数据标注
    elif x==3:#数据集拉取 拉取哪个表的数据集
        operator="AcquireDataSet"
        operator=bytes(operator,'utf-8')
        message=struct.pack('>I', len(operator)) + operator
        table=input("请输入拉取数据集的表名：\n")
        table=bytes(table,'utf-8')
        message+=struct.pack('>I',len(table))+table
        images=[]
        labels=[]
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.connect((HOST,PORT))#建立连接
            s.sendall(message)#发送数据给服务端命令
            s.recv(1)
            #接受数据集 
            while True:
                Len=struct.unpack('>I',s.recv(4))[0]
                if Len==0xffffffff:break
                img=s.recv(Len)
                images.append(img)
                Len=struct.unpack('>I',s.recv(4))[0]
                string=s.recv(Len)
                string=string.decode('utf-8')
                labels.append(string)
            s.sendall(b'1')#确认号
            s.close()
        #images装 图片信息
        #labels装 字符串
        print("获取的数据集大小:",len(images))
        y=int(input("输入选取的训练方式：\n 1."))
        if y==1:
            OT.opencv_train(images,labels)
        elif y==2:
            CT.train(images,labels)
        elif y==3:
            FCT.FRCNN_train(images,labels)
        elif y==4:
            YL.train(images,labels)
    elif x==4:
        y=int(input("输入选用哪种方法识别：\n1：opencv\n2:cnn\n3:YOLO\n"))
        if y==1:
            OT.predict()
        elif y==2:
            CT.predict()
        elif y==3:
            YL.predict()
        
    elif x==5:#test
        message="aac#"
        binary_data= b'Hello, world!'
        message+=binary_data.decode('utf-8')
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.connect((HOST,PORT))#建立连接
           # 在使用`s.sendall()`发送字符串数据之前，需要将字符串转换为字节串（bytes）。
           # 因为`s.sendall()`函数只接受字节串数据。可以使用字符串的`encode()`方法将其转换为字节串。
            s.sendall(message.encode('utf-8'))#发送数据给服务端命令
            #data=s.recv(1024)#接受服务端返回的数据  是二进制bits流
            #44print('Received',repr(data.decode()))#打印收到的数据
            s.close()#客户端主动断开连接

    pass
# with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
#     s.connect((HOST,PORT))#建立连接
#     s.sendall(b'Hello,World!')#发送数据给服务端
#     data=s.recv(1024)#接受服务端返回的数据
#     print('Received',repr(data.decode()))#打印收到的数据
#     s.close()#客户端主动断开连接
