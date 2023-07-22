import csv
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from pylab import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

config = {
    "font.family": 'serif',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

tick_labels_style = {
    "fontname": "Times New Roman",
    "fontsize": 12,
}


class MyFigureCanvas(FigureCanvas):
    def __init__(self):
        # 画布上初始化一个图像
        self.figure = Figure()
        super().__init__(self.figure)

    def plot(self, y_data: List[List[float]], legends, colors, linestyles, xlabel, ylabel, csv_title='', lim_axis=False):
        if len(y_data) != len(legends) != len(colors) != len(linestyles):
            raise Exception('params length dont match')
        # 设置图片大小
        # plt.figure(figsize=(8, 6))
        x_data = range(1, len(y_data[0]) + 1)
        self.plt.title(csv_title)
        self.plt.xlabel(xlabel)
        self.plt.ylabel(ylabel)
        self.plt.xticks(**tick_labels_style)
        self.plt.yticks(**tick_labels_style)
        # 限定y轴范围
        if lim_axis:
            self.plt.ylim(0, 1)
        # x轴坐标显示为整数
        self.plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        for idx, data in enumerate(y_data):
            self.plt.plot(x_data, data, label=legends[idx], linewidth=1, color=colors[idx], linestyle=linestyles[idx])

        self.plt.legend()
        # plt.show()
