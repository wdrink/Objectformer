import torch
import torch.nn as nn

from ObjectFormer.utils.registries import MODEL_REGISTRY

'''
MODEL:
  MODEL_NAME: ResNet50
  PRETRAINED: True
'''
REPO_OR_DIR = 'pytorch/vision:v0.10.0'


class BaseResNet(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.net = nn.Sequential(
            torch.hub.load(
                REPO_OR_DIR, self.net_name, **self.kws
            ),
            nn.Linear(1000, model_cfg['NUM_CLASSES']),
        )

    def forward(self, samples):
        x = samples['img']
        mask = torch.rand(x.shape[0],x.shape[2],x.shape[3], 2).cuda()
        return {'logits': self.net(x), 'mask':mask}


@MODEL_REGISTRY.register()
class ResNet18(BaseResNet):
    def __init__(self, model_cfg) -> None:
        self.net_name = 'resnet18'
        self.kws = {}
        if model_cfg['PRETRAINED']:
            self.kws = {'weights':'ResNet18_Weights.IMAGENET1K_V1'}

        super().__init__(model_cfg)


@MODEL_REGISTRY.register()
class ResNet34(BaseResNet):
    def __init__(self, model_cfg) -> None:
        self.net_name = 'resnet34'
        self.kws = {}
        if model_cfg['PRETRAINED']:
            self.kws = {'weights':'ResNet34_Weights.IMAGENET1K_V1'}
        super().__init__(model_cfg)


@MODEL_REGISTRY.register()
class ResNet50(BaseResNet):
    def __init__(self, model_cfg) -> None:
        self.net_name = 'resnet50'
        self.kws = {}
        if model_cfg['PRETRAINED']:
            self.kws = {'weights':'ResNet50_Weights.IMAGENET1K_V1'}
        super().__init__(model_cfg)


@MODEL_REGISTRY.register()
class ResNet101(BaseResNet):
    def __init__(self, model_cfg) -> None:
        self.net_name = 'resnet101'
        self.kws = {}
        if model_cfg['PRETRAINED']:
            self.kws = {'weights':'ResNet101_Weights.IMAGENET1K_V1'}
        super().__init__(model_cfg)


@MODEL_REGISTRY.register()
class ResNet152(BaseResNet):
    def __init__(self, model_cfg) -> None:
        self.net_name = 'resnet152'
        self.kws = {}
        if model_cfg['PRETRAINED']:
            self.kws = {'weights':'ResNet152_Weights.IMAGENET1K_V1'}
        super().__init__(model_cfg)
