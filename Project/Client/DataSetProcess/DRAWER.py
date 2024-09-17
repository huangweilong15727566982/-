import cv2
from config import *
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QMessageBox,QDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
class RectangleDrawer:
    def __init__(self, img):
        self.image = img
        self.rectangles = []
        self.last_image=[]
        self.move_image=img
        self.drawing = False
        self.last_rectangle = None
        self.label=[]
        cv2.imshow("press 'q' to exit", self.image)
        cv2.setMouseCallback("press 'q' to exit", self.mouse_event_handler)
    def mouse_event_handler(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:#记录起点
            self.drawing = True
            self.last_rectangle = None
            self.startpos=(x,y)
        elif event == cv2.EVENT_MOUSEMOVE:#鼠标移动过程并不绘制 只是显示一个变化的矩形框
            if self.drawing:
            
                    # cv2.rectangle(self.image, self.last_rectangle[0], self.last_rectangle[1], (255, 255, 255), -1)
                self.move_img=self.image.copy()
                cv2.rectangle(self.image, (self.startpos[0], self.startpos[1]), (x, y), (0, 0, 255), 2)
                #self.last_rectangle = [(self.startpos[0], self.startpos[1]), (x, y)]
                cv2.imshow("press 'q' to exit", self.image)
                self.image=self.move_img.copy()
        elif event == cv2.EVENT_LBUTTONUP:
            self.last_rectangle = [(self.startpos[0], self.startpos[1]), (x, y)]
            self.drawing = False
            self.last_image.append(self.image.copy())
            cv2.rectangle(self.image, (self.startpos[0], self.startpos[1]), (x, y), (0, 0, 255), 2)
            x=Label()
            self.label.append(x)
            self.rectangles.append(self.last_rectangle)

    def undo_last_rectangle(self):
        if len(self.rectangles) > 0:
            self.label.pop()
            self.last_rectangle = self.rectangles.pop()
            self.image=self.last_image.pop()
            cv2.imshow("press 'q' to exit", self.image)
    def run(self):
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):#结束绘制返回所有矩形框的图像
                cv2.destroyAllWindows()
                return self.rectangles,self.label
            elif key == ord("z"):#撤销上一个矩形框
                self.undo_last_rectangle()
        cv2.destroyAllWindows()
# img=cv2.imread(r"../images/1.jpg")
# # cv2.imshow("img",img)
# rect_drawer = RectangleDrawer(img)
# rect_drawer.run()