#%%
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import cirtorch.functional as LF
import math
import datasets.cub200 as dataset
import torch.nn.functional as F


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class Model(nn.Module):
    def __init__(self, opt=None, mu_dim=2048, sigma_dim=1, ):
        super().__init__()
        self.loss = opt.loss
        self.sigma_dim = 1
        self.var_init = 1e-3

        self.mu_dim =2047
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True, verbose=False)
        # resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.IMAGENET1K_V1', verbose=False)
        features = list(resnet50.children())[:-2]                                                  # feature map: ([B,3,224,224])->([B,2048,7,7])
        self.backbone = nn.Sequential(*features)

        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)

        self.mean_head = GeM()
        self.mean_fc = nn.Linear(2048, self.mu_dim)
        # self.mean_relu = nn.ReLU()
        self.l2n = L2Norm(dim=1)

        self.var_head = GeM()

        self.var_fc1 = nn.Linear(2048, 500)
        self.var_relu1 = nn.ReLU()
        self.var_fc2 = nn.Linear(500, self.sigma_dim, bias=True)
        self.var_fc2.weight.data.zero_()
        self.var_fc2.bias.data.copy_(torch.log(torch.tensor(self.var_init)))                       # NOTE: log?

        # self.var_fc1 = nn.Linear(2048, 2048)
        # self.var_relu1 = nn.ReLU()
        # self.var_fc2 = nn.Linear(2048, self.sigma_dim, bias=True)

        self.var_sp = nn.Softplus()

    def forward(self, inputs):
        B, C, H, W = inputs.shape             # (B, 1, 3, 224, 224)
        backbone_output = self.backbone(inputs)  # ([B, 2048, 7, 7])

        mu = self.mean_head(backbone_output).view(B, -1)                                           # ([B, 2048, 1, 1])
        mu = self.mean_fc(mu)                                                                      # ([B, 2047])
        # mu = self.mean_relu(mu)
        mu = self.l2n(mu)

        sigma = self.var_head(backbone_output).view(B, -1)                                         # ([B, 2048, 1, 1])
        sigma = self.var_relu1(self.var_fc1(sigma))                                                # ([B, 2048])
        sigma = self.var_fc2(sigma)                                                                # ([B, 1])
        sigma = self.var_sp(sigma)                                                                 # ([B, 1])

        mu_sampled = self.reparametrize_trick(mu, sigma)

        return mu_sampled, [mu, sigma]  # mu_hat, [mu, sigma^2]

    def reparametrize_trick(self, mu, sigma):
        sigma = sigma.sqrt()
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
