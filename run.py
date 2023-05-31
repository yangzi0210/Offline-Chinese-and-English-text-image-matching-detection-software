import sys
import os
from PyQt5.QtGui import QPixmap, QImage
from pyqt5_plugins.examplebuttonplugin import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsPixmapItem, QGraphicsScene
from PyQt5.QtCore import QObject, pyqtSignal
import txtimgui
import qdarkstyle
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # 只有直接运行这个脚本，才会往下执行
    # 别的脚本文件执行，不会调用这个条件句
    # 实例化，传参
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # 创建对象
    mainWindow = QMainWindow()
    # 创建ui，引用demo1文件中的Ui_MainWindow类
    ui = txtimgui.Ui_MainWindow()
    # 调用Ui_MainWindow类的setupUi，创建初始组件
    ui.setupUi(mainWindow)
    # 创建窗口
    mainWindow.show()

    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec_())