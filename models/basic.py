import time
import torch

from torch import nn


class BasicModule(nn.Module):
    """
    作为基类，继承 nn.Module 但增加了模型保存和加载功能 save and load
    """

    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))

    def load(self, model_path):
        """
        根据模型路径加载模型
        :param model_path: 模型路径
        :return: 模型
        """
        self.load_state_dict(torch.load(model_path))

    def save(self, filename=None):
        """
        保存模型，默认使用 "模型名字 + 时间" 作为文件名，也可以自定义
        如 ResNet34_20250710_23:57:29.pth
        """
        if filename is None:
            filename = 'checkpoints/' + self.model_name + '_' + time.strftime("%Y-%m-%d%H%M%S") + '.pth'
        torch.save(self.state_dict(), filename)
        return filename
