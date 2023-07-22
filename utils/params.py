import argparse


class Params:
    def __init__(self, widget):
        self.dataset = widget.datasets.text()
        self.model_name = widget.models.text()
        self.optimizer = widget.optimizer.text()
        self.global_epochs = int(widget.global_epochs.text())
        self.batch_size = int(widget.batch_size.text())
        self.lr = float(widget.lr.text())
        self.total_workers = int(widget.total_workers.text())
        self.global_lr = float(widget.global_lr.text())
        self.adversary_num = int(widget.adversary_num.text())
        self.local_epochs = int(widget.local_epochs.text())
        self.save_model = widget.save_model.isChecked()
        self.nb_classes = 10
        self.num_workers = 0
        self.data_path = '~/.torch'
        self.lambda_ = 0.01
        self.momentum = 0.0001
        # attack settings
        self.attack = widget.attack.isChecked()
        self.poisoning_rate = float(widget.poisoning_rate.text())
        self.trigger_label = int(widget.trigger_label.text())
        self.trigger_path = './backdoor/triggers/' + widget.trigger_path.text()
        self.trigger_size = int(widget.trigger_size.text())
        self.need_scale = widget.need_scale.isChecked()
        self.weight_scale = int(widget.weight_scale.text())
        epochs = list(range(40))
        self.attack_epochs = epochs[int(widget.attack_epochs.text()):]
        self.attack_method = 'central'
        # defense settings
        self.defense = widget.defense.isChecked()
        self.cluster = widget.cluster.isChecked()
        self.clip = widget.clip.isChecked()


def init_parser(description):
    parser = argparse.ArgumentParser(description=description)

    # base settings
    parser.add_argument('--dataset', default='MNIST',
                        help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')
    parser.add_argument('--model_name', default='badnets', help='[badnets, resnet18]')

    # 保存模型
    parser.add_argument('--load_local', action='store_true', help='使用保存的全局模型开始训练')
    parser.add_argument('--model_path', default='', help='保存的模型路径')
    parser.add_argument('--start_epoch', type=int, default=0, help='继续训练的全局轮次')

    parser.add_argument('--loss', default='mse')
    parser.add_argument('--optimizer', default='sgd', help='[sgd, adam]')
    parser.add_argument('--global_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2, help='')
    parser.add_argument('--data_path', default='~/.torch',
                        help='Place to load dataset (default: ~/.torch)')

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lambda_', type=float, default=0.01, help='')
    parser.add_argument('--momentum', type=float, default=0.0001, help='')

    # federated settings
    parser.add_argument('--total_workers', type=int, default=4)
    parser.add_argument('--global_lr', type=float, default=0.75)

    parser.add_argument('--adversary_num', type=int, default=1)
    parser.add_argument('--local_epochs', type=int, default=2)

    # 数据分布
    parser.add_argument('--no_iid', action='store_true')
    '''
    使用Dirichlet分布模拟no-iid，
    https://zhuanlan.zhihu.com/p/468992765
    '''
    parser.add_argument('--alpha', type=float, default=0.1)

    return parser
