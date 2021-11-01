import torch
import torch.nn
import numpy as np
import cv2
from vggnet import *
import time

device = torch.device('cuda')
model = VGG_net(in_channels=3, num_classes=3)
model = model.to(device)

PATH="./weights/"
model.load_state_dict(torch.load(PATH+'model.pt'))

classes = ['right','left','nothing']


img = cv2.imread('WIN_20211008_10_36_51_Pro 262.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
dst = np.transpose(dst, (2, 0, 1))
tensor = torch.Tensor(dst)
tensor = tensor.unsqueeze(0)
images = tensor.cuda()
outputs = model(images)
print(outputs)
_, predicted = torch.max(outputs, 1)

text = classes[predicted.item()]

print(text)
# cv2.imshow('test',img)
# time.sleep(0.4)
# if cv2.waitKey(1) == 27:
#     break
# print(f'prediction: {text}') 

