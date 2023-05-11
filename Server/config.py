import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QMessageBox,QDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
import tkinter as tk
from tkinter import simpledialog
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

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = App()
#     if ex.exec_() == QDialog.Accepted:
#         print(ex.textbox.text()) # 打印文本框的值
#     sys.exit()
def Label():
    root = tk.Tk()
    root.withdraw()

    user_input = None

# 使用循环，直到用户输入整数为止
    while user_input is None:
        user_input = simpledialog.askinteger(title="Label",
                                         prompt="输入标签类型号:")
    return user_input