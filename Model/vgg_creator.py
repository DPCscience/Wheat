import torch
from torchvision import models

vgg16 = models.vgg16(pretrained=True)
dic = vgg16.state_dict()
torch.save(dic, "vgg_16.model_dict")
