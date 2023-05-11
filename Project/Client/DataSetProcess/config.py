import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QMessageBox,QDialog,QFileDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
import cv2
import numpy as np
class App(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('输入标签号')
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(100, 100, 150, 50)
        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()
        self.label = QLabel('标签号:')
        self.label.setStyleSheet('font-size: 12px;')
        vbox.addWidget(self.label)

        self.textbox = QLineEdit(self)
        self.textbox.setValidator(QtGui.QIntValidator()) # 限制只能输入整数
        self.textbox.setFixedWidth(100) # 将输入框宽度设置为100
        self.textbox.returnPressed.connect(self.return_input) # 按下“Enter”键时返回文本框的值并退出窗口
        vbox.addWidget(self.textbox)

        self.setLayout(vbox)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def return_input(self):
        try:
            value = int(self.textbox.text())
            self.accept() # 返回文本框的值并退出窗口
        except ValueError:
            QMessageBox.warning(self, '错误', '请输入整数！')
            # 显示错误提示框并清除输入框
            self.textbox.clear()
            self.textbox.setFocus()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragPos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.dragPos)
            event.accept()
def Label():
    root = tk.Tk()
    root.withdraw()

    user_input = None

# 使用循环，直到用户输入整数为止
    while user_input is None:
        user_input = simpledialog.askinteger(title="Label",
                                         prompt="输入标签类型号:")
    return user_input
def FileFolderChoose():#文件夹选择
    folder_path=QFileDialog.getExistingDirectory(
        None,"选择文件夹","",QFileDialog.ShowDirsOnly
    )
    return folder_path
def notice(table):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo('Notice', f'请标注{table}')
def fileChoose():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

# print(fileChoose())
#缩放图片大小
def resize_image(image, new_size):
    # 获取原始图像的宽度和高度
    h, w = image.shape[:2]
    #print(h,w)
    # 计算缩放比例
    ratio = min(float(new_size[0])/w, float(new_size[1])/h)
    # 缩放图像
    resized_image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    # 创建填充后的画布
    canvas = 0 * np.ones((new_size[1], new_size[0], 3), dtype=np.uint8)
    h_, w_ = resized_image.shape[:2]
    # 将图像置于填充画布左上角
    x = 0
    y = 0
    canvas[y:y+h_, x:x+w_, :] = resized_image[:, :, :]
    return canvas