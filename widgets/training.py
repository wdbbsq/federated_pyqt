from typing import List

import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QValidator, QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QFrame, QLabel, QLineEdit, QGridLayout, QTextEdit, QHBoxLayout, QVBoxLayout, QWidget, \
    QLayout
from qfluentwidgets import NavigationWidget, ComboBox, SpinBox, LineEdit, PrimaryPushButton
from qfluentwidgets import SwitchButton
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import models
from run_backdoor import boot_backdoor
from utils.params import Params


class TrainingWidget(QFrame):
    """
    训练模型的 Widget
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # 全局布局
        global_layout = QGridLayout()
        # 局部布局
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        '''
        base setting
        '''
        base_layout = QGridLayout()
        base_layout.setSpacing(20)
        row = 0

        self.datasets = self.get_combobox(['MNIST', 'CIFAR10'])
        row = add_widgets(base_layout, '数据集', self.datasets, row)

        self.models = self.get_combobox(['badnets', 'resnet18'])
        row = add_widgets(base_layout, '模型', self.models, row)

        self.optimizer = self.get_combobox(['sgd', 'adam'])
        row = add_widgets(base_layout, 'optimizer', self.optimizer, row)

        # todo change back to 100
        self.global_epochs = self.get_lineedit(QIntValidator(self), 10, 120, '2')
        row = add_widgets(base_layout, '全局训练轮次', self.global_epochs, row)

        self.batch_size = self.get_lineedit(QIntValidator(self), 10, 100, '64')
        row = add_widgets(base_layout, '批次大小', self.batch_size, row)

        self.lr = self.get_lineedit(QDoubleValidator(self), 0.0001, 1, '0.01')
        row = add_widgets(base_layout, '学习率', self.lr, row)

        self.total_workers = self.get_lineedit(QIntValidator(self), 2, 30, '4')
        row = add_widgets(base_layout, '客户端总数', self.total_workers, row)

        self.global_lr = self.get_lineedit(QDoubleValidator(self), 0.001, 1, '0.75')
        row = add_widgets(base_layout, '全局学习率', self.global_lr, row)

        self.adversary_num = self.get_lineedit(QIntValidator(self), 0, 10, '1')
        row = add_widgets(base_layout, '恶意客户端数量', self.adversary_num, row)

        self.local_epochs = self.get_lineedit(QIntValidator(self), 1, 10, '2')
        row = add_widgets(base_layout, '本地迭代次数', self.local_epochs, row)

        self.save_model = self.get_switch()
        row = add_widgets(base_layout, '是否保存模型', self.save_model, row)

        '''
        poison settings
        '''
        attack_layout = QGridLayout()
        attack_layout.setSpacing(20)
        row = 0

        self.attack = self.get_switch()
        row = add_widgets(attack_layout, '是否攻击', self.attack, row)

        self.poisoning_rate = self.get_lineedit(QDoubleValidator(self), 0, 1, '0.1')
        row = add_widgets(attack_layout, '投毒比例', self.poisoning_rate, row)

        self.trigger_label = self.get_lineedit(QIntValidator(self), 0, 10, '1')
        row = add_widgets(attack_layout, '目标标签', self.trigger_label, row)

        self.trigger_path = self.get_combobox(['trigger_10', 'trigger_white'])
        row = add_widgets(attack_layout, '触发器样式', self.trigger_path, row)

        self.trigger_size = self.get_lineedit(QIntValidator(self), 0, 20, '5')
        row = add_widgets(attack_layout, '触发器大小', self.trigger_size, row)

        self.need_scale = self.get_switch()
        row = add_widgets(attack_layout, '是否进行缩放', self.need_scale, row)

        self.weight_scale = self.get_lineedit(QIntValidator(self), 1, 100, '100')
        row = add_widgets(attack_layout, '缩放大小', self.weight_scale, row)

        self.attack_epochs = self.get_lineedit(QIntValidator(self), 0, 40, '29')
        row = add_widgets(attack_layout, '进行攻击的轮次', self.attack_epochs, row)

        '''
        defense settings
        '''
        defense_layout = QGridLayout()
        defense_layout.setSpacing(20)
        row = 0

        self.defense = self.get_switch()
        row = add_widgets(defense_layout, '是否防御', self.defense, row)

        self.cluster = self.get_switch()
        row = add_widgets(defense_layout, '是否使用聚类', self.cluster, row)

        self.clip = self.get_switch()
        row = add_widgets(defense_layout, '是否使用剪枝', self.clip, row)

        '''
        LEFT ZONE
        '''
        left_box = QWidget(self)
        left_box.setLayout(left_layout)

        # basic settings
        base_box = QWidget(left_box)
        base_box.setLayout(base_layout)

        # attack settings
        attack_box = QWidget(left_box)
        attack_box.setLayout(attack_layout)

        # defense settings
        defense_box = QWidget(left_box)
        defense_box.setLayout(defense_layout)

        # Add to left
        left_layout.addWidget(base_box)
        left_layout.addWidget(attack_box)
        left_layout.addWidget(defense_box)

        self.train_btn = PrimaryPushButton('开始训练', self)
        self.train_btn.clicked.connect(self.start_training)
        left_layout.addWidget(self.train_btn, alignment=Qt.AlignCenter)
        left_layout.addStretch(1)

        '''
        RIGHT ZONE
        '''
        right_box = QWidget(self)
        right_box.setLayout(right_layout)


        global_layout.addWidget(left_box, 0, 0)
        global_layout.addWidget(right_box, 0, 1)

        self.setLayout(global_layout)

    def start_training(self):
        """
        开始训练
        """
        # Disable button
        # self.train_btn.setEnabled(False)

        # Construct params
        args = Params(self)
        print('Params inited.')

        # Start training
        # prefix = boot_backdoor(args)
        prefix = ''



        # Load model
        model = models.get_model('badnets')
        model.load_state_dict(torch.load('./saved_models/2023-05-29-10-42-40.pth.tar'))
        print('Model loaded.')

        # Draw pic


        # Enable button
        # self.train_btn.setEnabled(True)

    def get_combobox(self, options: List):
        """
        下拉选择框
        """
        combobox = ComboBox(self)
        combobox.addItems(options)
        combobox.setCurrentIndex(0)
        return combobox

    def get_lineedit(self, validator: QValidator, min_val, max_val, default_val):
        """
        数值输入框
        """
        lineedit = LineEdit(self)
        lineedit.setText(self.tr(default_val))
        validator.setRange(min_val, max_val)
        lineedit.setValidator(validator)
        return lineedit

    def get_switch(self):
        """
        开关
        """
        switch = SwitchButton(parent=self)
        switch.setText('')
        return switch


def add_widgets(layout: QLayout, label: str, widget: QWidget, row: int):
    """
    添加 QLabel 和对应的组件
    """
    layout.addWidget(QLabel(label), row / 2, 2 * (row % 2), alignment=Qt.AlignRight)
    layout.addWidget(widget, row / 2, 2 * (row % 2) + 1, alignment=Qt.AlignLeft)
    return row + 1
