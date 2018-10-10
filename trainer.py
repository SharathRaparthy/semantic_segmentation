import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def trainer(net,optimizer, data_loader,num_epoch):

     criterion = nn.CrossEntropyLoss()
     for epoch in range(num_epoch):
         for step, batch in enumerate(data_loader,0):
             images, labels = batch
             images = Variable(images).float().cuda()
             labels = Variable(labels).float().cuda()
             model = net.float().cuda()
             outputs = model(images)
             outputs = outputs.long().cuda()
             loss = criterion(outputs, labels)
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()

             if (step + 1)% 100 == 0:
                 print("The loss after %d steps in %d epochs is"% step % epoch, loss.item())
                 torch.save(model, '/home/sharath/semantic_segmentation/model.pth')
             if epoch%2 == 0:
                 torchvision.utils.save_image(outputs,'/home/sharath/semantic_segmentation/prediction.bmp')
                 torchvision.utils.save_image(labels,'/home/sharath/semantic_segmentation/labels.bmp')
                 torchvision.utils.save_image(images,'/home/sharath/semantic_segmentation/input.bmp')
