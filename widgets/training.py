from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLabel, QLineEdit, QGridLayout, QTextEdit, QHBoxLayout, QVBoxLayout, QWidget, \
    QLayout
from qfluentwidgets import NavigationWidget, ComboBox
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

        # base setting
        base_layout = QGridLayout()
        base_layout.setSpacing(20)
        row = 0

        datasets = ComboBox(self)
        datasets.addItems(['MNIST', 'CIFAR10'])
        datasets.setCurrentIndex(0)
        # datasets.currentTextChanged.connect(print)
        add_widgets(base_layout, '数据集', datasets, row)
        # base_layout.addWidget(SwitchButton(parent=self), 0, 1, alignment=Qt.AlignLeft)
        row += 1

        models = ComboBox(self)
        models.addItems(['badnets', 'resnet18'])
        models.setCurrentIndex(0)
        add_widgets(base_layout, '模型', models, row)
        row += 1

        optimizer = ComboBox(self)
        optimizer.addItems(['sgd', 'adam'])
        optimizer.setCurrentIndex(0)
        add_widgets(base_layout, 'optimizer', optimizer, row)
        row += 1

        # right

        left_box = QWidget(self)
        right_box = QWidget(self)
        left_box.setLayout(left_layout)
        right_box.setLayout(right_layout)

        base_box = QWidget(left_box)
        base_box.setLayout(base_layout)

        left_layout.addWidget(base_box)
        left_layout.addStretch(1)

        global_layout.addWidget(left_box, 0, 0)
        global_layout.addWidget(right_box, 0, 1)

        self.setLayout(global_layout)

        # base settings

        # poison settings

        # defense settings

        # self.label = QLabel('数据集', self)
        # self.label.setAlignment(Qt.AlignCenter)
        # self.hBoxLayout = QHBoxLayout(self)
        # self.hBoxLayout.addWidget(self.label, 1, Qt.AlignCenter)
        #
        # self.switchButton = SwitchButton(parent=self)
        # self.switchButton.move(48, 24)
        # self.switchButton.checkedChanged.connect(self.onCheckedChanged)

    def onCheckedChanged(self, is_checked: bool):
        text = 'On' if is_checked else 'Off'
        self.dataset_btn.setText(text)


def add_widgets(layout: QLayout, label: str, widget: QWidget, row: int):
    """
    添加 QLabel和对应的组件
    :param layout:
    :param label:
    :param widget:
    :param row:
    :return:
    """
    layout.addWidget(QLabel(label), row / 2, 2 * (row % 2), alignment=Qt.AlignRight)
    layout.addWidget(widget, row / 2, 2 * (row % 2) + 1, alignment=Qt.AlignLeft)
