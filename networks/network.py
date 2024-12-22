#%%
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import cirtorch.functional as LF
import torch.nn.functional as F
import warnings


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

def reparametrize_trick(mu, sigma):
    '''
    mu: [B, mu_dim]
    sigma: [B, sigma_dim] (meaning: sigma^2)
    return: [B, mu_dim]
    '''
    sigma = sigma.sqrt()
    eps = torch.randn_like(sigma)
    return mu + eps * sigma


class Model(nn.Module):
    def __init__(self, mu_dim=2048, sigma_dim=1, sigma_init=1e-3, dropout_rate=0.1, setting='btl'):                   # setting='btl(btl)', 'dul(triplet+kl)', 'mcd(triplet)', 'triplet(triplet)'
        super().__init__()
        self.setting = setting
        self.mu_dim =mu_dim
        self.sigma_dim = sigma_dim

        # backbone
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resnet50 = torch.hub.load('pytorch/vision:v0.10.0',
                                      'resnet101',
                                      pretrained=True,
                                      verbose=False)
        features = list(resnet50.children())[:-2]                                                  # feature map: ([B,3,224,224])->([B,2048,7,7])
        self.backbone = nn.Sequential(*features)
        # freeze BN
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
        # mean head
        self.mean_head = nn.Sequential(
            GeM(),
            nn.Flatten(),
            nn.Linear(2048, mu_dim),
            L2Norm(dim=1),
        )

        # sigma head
        if self.setting in ['btl', 'dul']:
            self.sigma_head = nn.Sequential(
                GeM(),
                nn.Flatten(),
                nn.Linear(2048, 500),
                nn.ReLU(),
                nn.Linear(500, sigma_dim, bias=True),
                nn.Softplus(),
            )
            self.sigma_head[-2].weight.data.zero_()
            self.sigma_head[-2].bias.data.copy_(torch.log(torch.tensor(sigma_init)))                          # NOTE: log?
        elif self.setting in ['mcd', 'triplet']:
            self.sigma_head = None
            if self.setting in ['mcd']:
                for module in self.backbone.modules():
                    if isinstance(module, nn.Conv2d):
                        module.register_forward_hook(lambda m, inp, out: F.dropout(
                            out,
                            p=dropout_rate,
                            training=True,    # m.training
                        ))

    def forward(self, inputs):
        feature_maps = self.backbone(inputs)  # ([B, 2048, 7, 7])
        mu = self.mean_head(feature_maps)                                           # ([B, 2048, 1, 1])

        if self.setting in ['btl', 'dul']:
            sigma = self.sigma_head(feature_maps)                          # ([B, 2048, 1, 1])
            if self.setting == 'btl':
                return mu, sigma
            elif self.setting == 'dul':
                return reparametrize_trick(mu, sigma), [mu, sigma]
        elif self.setting in ['mcd', 'triplet']:
            return mu, torch.zeros((mu.shape[0], self.sigma_dim), device=mu.device)


#%%

if __name__ == '__main__':
    model = Model(setting='dul')
    output = model(torch.zeros((8,3,224,224)))
    print(output)
