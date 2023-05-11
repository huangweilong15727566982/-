import socket
import threading
import numpy as np
from DataBase import DB
import struct
import cv2
host = '10.251.197.57' # 服务器的IP地址
port = 12345 # 服务器的端口号
# 处理客户端连接请求
def handle_client(sock, addr):
    # 处理连接请求的代码
    print(f'New connection from {addr[0]}:{addr[1]}')
    with sock:

        Len = struct.unpack('>I',sock.recv(4)) # 接收操作码长度
        print(Len)
        operator=sock.recv(Len[0])#获取字节流
        sock.sendall(b"1")#确认
        operator=operator.decode('utf-8')#解码为字符串
        print(operator)
        if not operator:
            print("命令不正确") # 连接断开
            return 
        # 处理接收到的数据
        print(f'Received data from {addr[0]}:{addr[1]}: {operator}')
        if operator=="ImportTask":#将任务导入数据库
            Len=struct.unpack('>I',sock.recv(4))[0]
            table=sock.recv(Len)
            table=table.decode('utf-8')
            images=[]
            while True:
                data=sock.recv(4)
                print("data:",data)
                Len = struct.unpack('>I',data)[0]
                if Len ==0xffffffff:
                    sock.sendall(b'1')
                    print("消息接受完毕！")
                    break
                img=sock.recv(Len)
                images.append(img)
                print("收到一张图片")
                # img_array=np.frombuffer(img,np.uint8)
                # img1=cv2.imdecode(img_array,cv2.IMREAD_COLOR)
                # cv2.imshow('img',img1)
                # cv2.waitKey(0)
           
            DB.addTaskSet(table,images) #将任务加入数据库
            cv2.destroyAllWindows()
        elif operator=="AcquireTask":#获取任务包
            message=DB.AcquireTask()
            sock.sendall(message)
            sock.recv(1)#等待确认
        elif operator=="CommitDataSet":#添加一张图片进数据集
            #DataSet=[]
            Len = struct.unpack('>I',sock.recv(4))[0]#接收图片长度
            #提交数据集的数据形式 图片长度（长度为0xffffffff表示结尾）+图片数据 +表长度+表+字符串的长度+字符串标签数据
            img=sock.recv(Len)#接受图片数据
            Len = struct.unpack('>I',sock.recv(4))[0]#接收表长度
            #获取表名
            table_name=sock.recv(Len)
            table_name=table_name.decode('utf-8')
            #获取字符串
            Len=struct.unpack('>I',sock.recv(4))[0]
            string=sock.recv(Len)
            sock.recv(4)#结束确认
            sock.send(b'1')#发送确认
            #DataSet.append((id,img))
            string=string.decode('utf-8')
            print("string:",string)
            DB.AddOnePic(img,string,table_name)
            print("写入数据库成功")   
        elif operator=="AcquireDataSet":#获取哪个表的数据集
            Len = struct.unpack('>I',sock.recv(4))[0]#接收表长度
            table=sock.recv(Len)
            table=table.decode('utf-8')
            DB.AcquireDataSet(sock,table)
            #sock.sendall(message)
            #服务器要确认客户端收到消息
            #sock.recv(1)
            #print("message",message)
        elif operator=="GetTableInfo":# 获取表信息
            message=DB.GetTableInfo()
            sock.sendall(message)
            sock.recv(1)#等待客户端的确认
        elif operator=="TestConection":# 测试连接
            sock.sendll(b'1')
        
        #     pass
# 建立服务器套接字
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, port))
    s.listen(10) # 最大连接数为10
    print(f'Server started on {host}:{port}')
    while True:#服务器保持打开
        conn, addr = s.accept() # 接受连接请求
        # 创建新线程处理连接请求
        t = threading.Thread(target=handle_client, args=(conn, addr))
        t.start()