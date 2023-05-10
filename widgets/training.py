from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QValidator, QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QFrame, QLabel, QLineEdit, QGridLayout, QTextEdit, QHBoxLayout, QVBoxLayout, QWidget, \
    QLayout
from qfluentwidgets import NavigationWidget, ComboBox, SpinBox, LineEdit, PrimaryPushButton
from qfluentwidgets import SwitchButton


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

        self.global_epochs = self.get_lineedit(QIntValidator(self), 10, 120, '100')
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

        self.trigger_path = self.get_combobox(['./trigger_10', './trigger_white'])
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

        # right

        left_box = QWidget(self)
        right_box = QWidget(self)
        left_box.setLayout(left_layout)
        right_box.setLayout(right_layout)

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

        train_btn = PrimaryPushButton('开始训练', self)
        train_btn.clicked.connect(self.start_training)
        left_layout.addWidget(train_btn, alignment=Qt.AlignCenter)
        left_layout.addStretch(1)

        global_layout.addWidget(left_box, 0, 0)
        global_layout.addWidget(right_box, 0, 1)

        self.setLayout(global_layout)

    def start_training(self):
        """
        开始训练
        """
        # construct params
        args = dict()
        # basic settings
        args.dataset = self.datasets.text()
        args.models = self.models.text()
        args.optimizer = self.optimizer.text()
        args.global_epochs = int(self.global_epochs.text())
        args.batch_size = int(self.batch_size.text())
        args.lr = float(self.lr.text())
        args.total_workers = int(self.total_workers.text())
        args.global_lr = float(self.global_lr.text())
        args.adversary_num = int(self.adversary_num.text())
        args.local_epochs = int(self.local_epochs.text())
        args.save_model = self.save_model.isChecked()
        # attack settings
        args.attack = self.attack.isChecked()
        args.poisoning_rate = float(self.poisoning_rate.text())
        args.trigger_label = int(self.trigger_label.text())
        args.trigger_path = self.trigger_path.text()
        args.trigger_size = int(self.trigger_size.text())
        args.need_scale = self.need_scale.isChecked()
        args.weight_scale = int(self.weight_scale.text())
        epochs = list(range(40))
        args.attack_epochs = epochs[int(self.attack_epochs.text()):]
        # defense settings


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
