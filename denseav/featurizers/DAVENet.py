# Author: David Harwath
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as imagemodels


class Davenet(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(Davenet, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1, 11), stride=(1, 1), padding=(0, 5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8))
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8))
        self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.squeeze(2)
        return x


class Resnet18(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet18, self).__init__(imagemodels.resnet.BasicBlock, [2, 2, 2, 2])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet18']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x


class Resnet34(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet34, self).__init__(imagemodels.resnet.BasicBlock, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet34']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x


class Resnet50(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet50, self).__init__(imagemodels.resnet.Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet50']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x


class VGG16(nn.Module):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(VGG16, self).__init__()
        seed_model = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        seed_model = nn.Sequential(*list(seed_model.children())[:-1])  # remove final maxpool
        last_layer_index = len(list(seed_model.children()))
        seed_model.add_module(str(last_layer_index),
                              nn.Conv2d(512, embedding_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.image_model = seed_model

    def forward(self, x):
        x = self.image_model(x)
        return x


def prep(dict):
    return {k.replace("module.", ""): v for k, v in dict.items()}


class DavenetAudioFeaturizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.audio_model = Davenet()
        self.audio_model.load_state_dict(prep(torch.load("../models/davenet_pt_audio.pth")))

    def forward(self, audio, include_cls):
        patch_tokens = self.audio_model(audio).unsqueeze(2)

        if include_cls:
            return patch_tokens, None
        else:
            return patch_tokens

    def get_last_params(self):
        return []


class DavenetImageFeaturizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.image_model = VGG16()
        self.image_model.load_state_dict(prep(torch.load("../models/davenet_pt_image.pth")))

    def forward(self, image, include_cls):
        patch_tokens = self.image_model(image)

        if include_cls:
            return patch_tokens, None
        else:
            return patch_tokens

    def get_last_params(self):
        return []
