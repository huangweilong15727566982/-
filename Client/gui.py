import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QObject, pyqtSignal
import cv2
import struct 
import socket
from TrainMethod import OpencvTrain as OT
from TrainMethod import CnnTrain as CT
from TrainMethod import FasterRCNN as FCT
from TrainMethod.YoloV3 import yolo_v3_learn as YL
import sys
from DataSetProcess import DataSetProcess as DSP
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
#         # 将标签添加到垂直布局中
#         layout = QVBoxLayout()
#         layout.addWidget(label)

#         # 设置小部件的布局
#         self.setLayout(layout)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('./images/icon.png'))
        self.HOST=""
        self.PORT=1
        self.resize(1920, 1080)  # 设定窗口大小\
        self.setStyleSheet("QMainWindow{background-image: url(./images/main_bg2);}")
        self.setWindowTitle('目标识别系统')
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # 添加服务器IP地址和端口号输入框
        self.server_label = QLabel('服务器地址：', self.central_widget)
        self.server_label.move(20, 30)
        self.server_HOST = QLineEdit(self.central_widget)
        self.server_HOST.setText("10.252.36.75")
        self.server_HOST.move(110, 20)
        self.server_HOST.resize(200, 30)
        self.server_HOST.setStyleSheet('background-color: rgba(255, 255, 255, 100);')
        self.server_PORT = QLineEdit(self.central_widget)
        self.server_PORT.setText("12345")
        self.server_PORT.move(320, 20)
        self.server_PORT.resize(50, 30)
        self.server_PORT.setStyleSheet('background-color: rgba(255, 255, 255, 100);')

        
        #添加测试连接按钮
        self.testconectionbnt=QPushButton('测试连接', self.central_widget)
        self.testconectionbnt.move(400,20)
        self.testconectionbnt.clicked.connect(self.TestConnection) 
        self.testconectionbnt.setStyleSheet('''
            QPushButton {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 10px;
                border-color: white;
                font: bold 15px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        # 添加文本显示框
        self.text_display = QTextEdit(self.central_widget)
        self.text_display.setReadOnly(True)
        self.text_display.move(20, 70)
        self.text_display.resize(416, 832)
        self.text_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.text_display.setStyleSheet('background-color: rgba(255, 125, 125, 25);')
        #添加文本显示框
        label = QLabel("临时按钮", self)
        # 设置标签的字体和颜色
        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        label.setFont(font)
        label.setStyleSheet("color: #672355")
        # 将标签移动到指定位置
        label.resize(150,75)
        label.move(500, 100)
        # 添加图像/视频播放框
        self.media_display = QLabel( self.central_widget)
        self.media_display.move(712, 70)  # 根据需求调整位置
        self.media_display.resize(832, 832)  # 根据需求修改大小
        self.media_display.setAlignment(Qt.AlignCenter)
        #self.media_display.setStyleSheet('background-color:black;')  # 根据需求修改颜色
        self.media_display.setStyleSheet('background-color: rgba(255, 125, 125, 25); border: 4px solid #ecd8d4; border-radius: 15px;padding: 10px; font-family: Arial; font-size: 24px;color:#d7b26c')
      

        # # 绘制电视机所有的直线和圆角矩形
        # pen = painter.pen()
        # pen.setColor(QColor('#ffffff'))
        # pen.setWidth(8)
        # painter.setPen(pen)
        # painter.drawRoundedRect(pixmap.rect(), 30, 30, Qt.AbsoluteSize)
        # painter.drawLine(0, 128, 0, pixmap.height() - 128)
        # painter.drawLine(pixmap.width() - 1, 128, pixmap.width() - 1, pixmap.height() - 128)
        # painter.drawLine(128, 0, pixmap.width() - 128, 0)
        # painter.drawLine(128, pixmap.height() - 1, pixmap.width() - 128, pixmap.height() - 1)
        # painter.drawEllipse(64, 64, 128, 128)

        # # 将上面绘制的图像设置为QLabel控件的背景
        # self.media_display.setPixmap(pixmap)
        # self.media_display.setAlignment(Qt.AlignCenter)
        # self.media_display.setStyleSheet('font-family: Arial; font-size: 24px; color: #ffffff;')
        # layout=self.layout()
        # layout.addWidget(self.media_display)

        # 添加四个按钮
        self.button1 = QPushButton('数据包导入', self.central_widget)
        self.button1.move(20, 925)
        self.button1.resize(150, 75)
        self.button1.clicked.connect(self.ImportTask)
        self.button1.setStyleSheet('''
            QPushButton {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 15px;
                border-color: white;
                font: bold 20px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        self.button2 = QPushButton('数据标注', self.central_widget)
        self.button2.move(190, 925)
        self.button2.resize(150, 75)
        self.button2.clicked.connect(self.DataAnnotation)
        self.button2.setStyleSheet('''
            QPushButton {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 15px;
                border-color: white;
                font: bold 20px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        self.button3 = QPushButton('训练模型', self.central_widget)
        self.button3.move(360, 925)
        self.button3.resize(150, 75)
        self.button3.clicked.connect(self.TrainModel)
        self.button3.setStyleSheet('''
            QPushButton {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 15px;
                border-color: white;
                font: bold 20px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        self.button4 = QPushButton('预测', self.central_widget)
        self.button4.move(530, 925)
        self.button4.resize(150, 75)
        self.button4.clicked.connect(self.Predict)
        self.button4.setStyleSheet('''
            QPushButton {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 15px;
                border-color: white;
                font: bold 20px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        # self.button5 = QPushButton('预测', self.central_widget)
        # self.button5.move(700, 925)
        # self.button5.resize(150, 75)
    def Back2MainPage(self):
        
        # # 显示图像
        self.media_display.setPixmap(QPixmap())
        
    def ShowCVImage(self,img):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        p=cv2qt(img)
        # 显示图像
        self.media_display.setPixmap(QPixmap.fromImage(p))
    def TestConnection(self): #测试连接
        self.text_display.append("--------测试连接--------")
        operator="TestConnection"
        operator=bytes(operator,'utf-8')
        message=struct.pack('>I', len(operator)) + operator
        try:
            self.HOST = self.server_HOST.text()
            self.PORT = int(self.server_PORT.text())
            print("HOST",self.HOST)
            print("PORT",self.PORT)
            with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
                s.connect((self.HOST,self.PORT))#建立连接
                s.sendall(message)#发送数据给服务端
                s.recv(1)
        except Exception as ex:
            print(111)
            self.text_display.append(str(ex))
            scroll_bar = self.text_display.verticalScrollBar()
            scroll_bar.setValue(scroll_bar.maximum())
            return
        self.text_display.append("连接成功")
        scroll_bar = self.text_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
    def ImportTask(self):#生成录制按钮
        self.text_display.append("--------数据包导入--------")
        operator="GetTableInfo"
        operator=bytes(operator,'utf-8')
        message=struct.pack('>I', len(operator)) + operator
        tables=[]#存储获取的表单信息
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.connect((self.HOST,self.PORT))#建立连接
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
        self.combo=QComboBox(self.central_widget)
        self.combo.move(500,270)
        self.combo.resize(150,75)
        layout = self.layout()
        layout.addWidget(self.combo) #新增控件使用
        self.text_display.append("选择需要导入数据包的表项\n点击开始按钮选择视频文件,不选则打开摄像头\n点击截获获取视频当前图片\n点击结束停止播放")
        scroll_bar = self.text_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        for item in tables:
            self.text_display.append(item)
            item=item.split()
            self.combo.addItem(item[1])
        self.combo.setStyleSheet('''
            QComboBox {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 15px;
                border-color: white;
                font: bold 20px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        self.startbutton = QPushButton('开始', self.central_widget)
        self.startbutton.move(500, 440)
        self.startbutton.resize(150, 75)
        self.startbutton.clicked.connect(self.StartImport)
        self.startbutton.setStyleSheet('''
            QPushButton {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 15px;
                border-color: white;
                font: bold 20px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        self.capturebutton = QPushButton('截获', self.central_widget)
        self.capturebutton.move(500, 610)
        self.capturebutton.resize(150, 75)
        self.capturebutton.clicked.connect(self.Capture)
        self.capturebutton.setStyleSheet('''
            QPushButton {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 15px;
                border-color: white;
                font: bold 20px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        self.capture=False
        self.endbutton = QPushButton('结束', self.central_widget)
        self.endbutton.move(500, 780)
        self.endbutton.resize(150, 75)
        self.endbutton.clicked.connect(self.End)
        self.endbutton.setStyleSheet('''
            QPushButton {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 15px;
                border-color: white;
                font: bold 20px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        self.end=False
        layout.addWidget(self.startbutton)
        layout.addWidget(self.capturebutton)
        layout.addWidget(self.endbutton)
    def End(self):
        self.end=True
    def StartImport(self):     
        table=self.combo.currentText()
        taskset=DSP.importTask(self)
        self.text_display.append(f"实际收集图片：{len(taskset)}")
        operator="ImportTask"#操作码
        operator=bytes(operator,'utf-8')
        table=bytes(table,'utf-8')
        message=struct.pack('>I', len(operator)) + operator
        message+=struct.pack('>I',len(table))+table
        mess=message
        #削减传输长度 每次最多传输4张图片
        end= 0xffffffff
        for img in taskset:
            mess+=struct.pack('>I', len(img)) + img
            mess+=struct.pack('>I',end)
            with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
                s.connect((self.HOST,self.PORT))#建立连接
                s.sendall(mess)#发送数据给服务端
                s.recv(1)#确认收到
                s.close()#客户端主动断开连接
            mess=message
        
        # with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
        #     s.connect((self.HOST,self.PORT))#建立连接
        #     s.sendall(mess)#发送数据给服务端
        #     s.recv(1)#确认收到
        #     s.close()#客户端主动断开连接 
        self.combo.deleteLater()
        self.startbutton.deleteLater()#销毁该状态的两个按钮
        self.endbutton.deleteLater()
        self.capturebutton.deleteLater()  
        self.Back2MainPage()#回到主页
    def Capture(self):
        self.capture=True
    def DataAnnotation(self):
        self.text_display.append("--------数据标注--------")
        operator="AcquireTask"
        operator=bytes(operator,'utf-8')
        message=struct.pack('>I', len(operator)) + operator
        #print(message.encode('utf-8').decode())
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.connect((self.HOST,self.PORT))#建立连接
            s.sendall(message)#发送数据给服务端命令
            s.recv(1)
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
             self.text_display.append("任务数据库中没有任务")
             DSP.notice("No Task!")#弹窗通知
        else:
             DSP.DataAnnotation(self,taskset,tables)#数据标注
        self.Back2MainPage()#返回主页
    def TrainModel(self):
        self.text_display.append("--------训练模型--------")
        #有哪些模型    opencv LBPH算法   CNN模型    YOLO模型
        self.text_display.append("请选择模型和数据集进行训练")
        self.combo1=QComboBox(self.central_widget)
        self.combo1.move(500,270)
        self.combo1.resize(150,75)
        self.combo1.addItem("opencv-LBPH算法")
        self.combo1.addItem("CNN算法")
        self.combo1.addItem("YOLO算法")
        self.combo2=QComboBox(self.central_widget)
        self.combo2.move(500,440)
        self.combo2.resize(150,75)
        #获取数据集表信息
        operator="GetTableInfo"
        operator=bytes(operator,'utf-8')
        message=struct.pack('>I', len(operator)) + operator
        tables=[]#存储获取的表单信息
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.connect((self.HOST,self.PORT))#建立连接
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
            item=item.split()
            self.combo2.addItem(item[1])
        self.starttrain = QPushButton('开始训练', self.central_widget)
        self.starttrain.move(500, 610)
        self.starttrain.resize(150, 75)
        self.starttrain.clicked.connect(self.StartTrain)
        layout = self.layout()
        layout.addWidget(self.combo1) #新增控件使用
        layout.addWidget(self.combo2)
        layout.addWidget(self.starttrain)
    def StartTrain(self):
        train_method=self.combo1.currentText()
        train_data=self.combo2.currentText()
        self.text_display.append(f"你选择了{train_method}和{train_data}数据集")
        operator="AcquireDataSet"
        operator=bytes(operator,'utf-8')
        message=struct.pack('>I', len(operator)) + operator
        table=train_data
        table=bytes(table,'utf-8')
        message+=struct.pack('>I',len(table))+table
        images=[]
        labels=[]
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.connect((self.HOST,self.PORT))#建立连接
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
        
        #创建图表 fig
        self.fig=Figure(figsize=(12,6),tight_layout=True)
        self.fig.suptitle('Training Metrics')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.move(712,70)
        self.canvas.resize(832,832)
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')

        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Accuracy')
        #将图标放入布局中
        layout=self.layout()
        layout.addWidget(self.canvas)
        self.text_display.append(f"获取的数据集大小:{len(images)}")
        self.text_display.ensureCursorVisible()  # 强制更新文本显示
        if train_method=="opencv-LBPH算法":
            OT.opencv_train(self,images,labels)
        elif train_method=="CNN算法":
            CT.train(self,images,labels)
        elif train_method=="YOLO算法":
            YL.train(self,images,labels)
        else : self.text_display.append("出错")
        #删除新增的控件
        self.combo1.deleteLater()
        self.combo2.deleteLater()
        self.starttrain.deleteLater()
        self.canvas.deleteLater()
        self.Back2MainPage()
        #print("获取的数据集大小:",len(images))
    def Predict(self):#
        #生成按钮
        self.text_display.append("--------检测目标--------")      
        self.combo=QComboBox(self.central_widget) #第一个下拉框
        self.combo.move(500,270)
        self.combo.resize(150,75)
        self.combo.addItem("face recognize")
        self.combo.addItem("object detection")
        self.combo.setStyleSheet('''
            QComboBox {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 15px;
                border-color: white;
                font: bold 20px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        
        layout = self.layout()
        layout.addWidget(self.combo)

        self.combo2=QComboBox(self.central_widget) #第二个下拉框
        self.combo2.move(500,345) #放在第一个下拉框下方
        self.combo2.resize(150,35)
        self.combo2.addItem("cnn") #初始化为不可见状态，等待用户选择"face recognize"或"object detection"后再显示
        self.combo2.addItem("opencv")
        self.combo2.addItem("yolo")
        self.combo2.setVisible(False) #不可见
        self.combo.activated.connect(self.on_combo_activated)  #连接到第一个下拉框选择事件
        self.combo2.setStyleSheet('''
            QComboBox {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 15px;
                border-color: white;
                font: bold 20px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        self.startbutton = QPushButton('开始', self.central_widget)
        self.startbutton.move(500, 440)
        self.startbutton.resize(150, 75)
        self.startbutton.setStyleSheet('''
            QPushButton {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 15px;
                border-color: white;
                font: bold 20px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        self.endbutton = QPushButton('停止', self.central_widget)
        self.endbutton.move(500, 610)
        self.endbutton.resize(150, 75)
        self.endbutton.clicked.connect(self.End)
        self.endbutton.setStyleSheet('''
            QPushButton {
                background-color: rgba(125, 125, 125, 125);
                border-style: outset;
                border-width: 1px;
                border-radius: 15px;
                border-color: white;
                font: bold 20px;
                color:  #672342;
                padding: 6px;
            }
        ''')
        self.end=False
        layout.addWidget(self.startbutton)
        layout.addWidget(self.endbutton)
        self.startbutton.clicked.connect(self.StartPredict)
    def on_combo_activated(self):  # 响应第一个下拉框的点击事件
        if self.combo.currentText() == "face recognize":  # 如果用户点击了 "face recognize"
            self.combo2.clear()  # 清空原来的选项
            self.combo2.addItem("cnn")
            self.combo2.addItem("opencv")
            self.combo2.addItem("yolo")
            self.combo2.setVisible(True)  # 显示第二个下拉框
        elif self.combo.currentText() == "object detection":  # 如果用户点击了 "object detection"
            self.combo2.clear()  # 清空原来的选项
            self.combo2.addItem("yolo")
            self.combo2.setVisible(True)  # 显示第二个下拉框

        # self.startbutton = QPushButton('开始', self.central_widget)
        # self.startbutton.move(500, 440)
        # self.startbutton.resize(150, 75)
        # self.startbutton.clicked.connect(self.StartPredict)
        # layout.addWidget(self.startbutton)

    def StartPredict(self):
        object=self.combo.currentText()
        method=self.combo2.currentText()
        self.text_display.append(f"你选择了{object}和{method}算法")
        if object=="face recognize":
              if method=="cnn":
                    CT.predict(self)
              elif method=="opencv":
                    OT.predict(self)
              else:
                  pass
        elif object=="object detection":
            if method=="yolo":
                YL.predict(self)
        else :self.text_display.append("出错了!")
        self.combo.deleteLater()
        self.combo2.deleteLater()
        self.startbutton.deleteLater()
        self.endbutton.deleteLater()
        self.Back2MainPage()

                
#将Opencv图像转为Qt格式
def cv2qt(img):
    h,w,ch=img.shape
    bytes_per_line=ch*w
    convert_to_Qt_format = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(main_window.media_display.width(), main_window.media_display.height(), Qt.KeepAspectRatio)
    return p

app = QApplication(sys.argv)
main_window = MainWindow()
main_window.Back2MainPage()
main_window.show()
sys.exit(app.exec_())
