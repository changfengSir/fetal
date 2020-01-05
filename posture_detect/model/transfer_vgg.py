import torch
import torchvision



model = torchvision.models.vgg16(pretrained=True)
# for param in model.parameters():
#     print(param)
# model.features = list(model.features.children())[:-1]

print(model.features)