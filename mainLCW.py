import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from torchinfo import summary
import numpy as np
import time
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

''' ** file structure setting
- project
    - data
        -places365
            - train
            - train_custom_100
            - train_custom_365
            - val
            - val_custom_100
            - val_custom_365
    - main.py
    - test.py
'''


# 1. Datasets preprocessing    ** option
'''
1. data augmentation
 : default = # transforms.RandomResizedCrop(224),
             # transforms.RandomHorizontalFlip(),
You can add more data augmentation technique and when you use, uncomment lines

2. normalize value - fixed
 : default = [0.485, 0.456, 0.406],  # ImageNet mean & variance
             [0.229, 0.224, 0.225]
It is used at ImageNet and usually used in many codes
'''
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean & variance
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.Lambda(lambda y: y.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

inverse_transforms = transforms.Compose([  # for image visualization
    transforms.Normalize(mean=[0., 0., 0.],
                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                         std=[1., 1., 1.]),
])


# 2. Datasets preparing    ** option
'''
1. dataset size
 : default = class_num = 365
             train_dir = 'data/Places365/train_custom_365'
             valid_dir = 'data/Places365/val_custom_365'
train data: 1.8 million (365 * 5000) -> 365000 (365 * 1000) -> 100000 (100 * 1000)
valid data: 3.6만개(365 * 100) -> 1.8만개(365 * 30) -> 0.3만개(100 * 30)

1.8M
    class_num = 365
    train_dir = 'data/Places365/train'
    valid_dir = 'data/Places365/val'

365k
    class_num = 365
    train_dir = 'data/Places365/train_custom_365'
    valid_dir = 'data/Places365/val_custom_365'
    
100k
    class_num = 100
    train_dir = 'data/Places365/train_custom_100'
    valid_dir = 'data/Places365/val_custom_100'

2. batch_size
 : default = 32
The larger dataset size, the larger batch size (min: 2 / max: 32)
'''

class_num = 365
train_dir = 'data/Places365/train_custom_365_2000'
valid_dir = 'data/Places365/val_custom_365_30'
batch_size = 32

data = {
    'train': datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_dir, transform=image_transforms['valid']),
}

train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)

train_size = len(data['train'])
valid_size = len(data['valid'])
print(' train data size:', train_size, ' valid data size:', valid_size)

# check you datasets
# train_data_iter = iter(train_data)
# inputs, labels = next(train_data_iter)  # inputs: [32, 3, 224, 224] / labels: [32]
#
# inputs_RGB = inverse_transforms(inputs)
# img = inputs_RGB[0].squeeze()
# label = labels[0]
# idx_to_class = os.listdir(train_dir)
#
# print("label number: {}, label: {}".format(label, idx_to_class[label]))
# plt.imshow(img.permute(1, 2, 0))
# plt.show()


# 3. Neural network & training setting      ** option
'''
1. FC_option (FC layer parameter setting )
 : default = 1
1 : one fully connected layer as classifier
2 : two fully connected layer with classifier
3 : three fully connected layer with classifier

2. freeze_level (trainable parameter at pre-trained network level setting)
 : default = 1
1 : only FC layer
2 : FC + 5_1
3 : FC + 5_1 + 4_1
4 : FC + 5_1 + 4_1 + 3_1
5 : FC + 5_1 + 4_1 + 3_1 + 2_1
6 : all
'''

FC_option = 1
freeze_level = 1

device = 'cuda'
iter_num = int(train_size / batch_size)
data_per_class_num = int(train_size / class_num)

vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

for param in vgg19.parameters():
    param.requires_grad = True

# FC option setting
if FC_option == 1:
    vgg19.classifier = nn.Sequential(  # Linear layer 추가 또는 앞의 conv layer 4_1, 5_1 training
        nn.Linear(25088, class_num)
    )
elif FC_option == 2:
    vgg19.classifier = nn.Sequential(  # Linear layer 추가 또는 앞의 conv layer 4_1, 5_1 training
        nn.Linear(25088, 1024, bias=True),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, class_num),
    )
elif FC_option == 3:
    vgg19.classifier = nn.Sequential(  # Linear layer 추가 또는 앞의 conv layer 4_1, 5_1 training
        nn.Linear(25088, 1024, bias=True),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024, bias=True),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, class_num),
    )
else:
    assert True, "FC_option is out of range!!"


# freeze_level setting
if freeze_level == 1:
    for param in vgg19.parameters():
        param.requires_grad = False
elif freeze_level == 2:
    for i in range(28):
        for param in vgg19.features[i].parameters():
            param.requires_grad = False
elif freeze_level == 3:
    for i in range(19):
        for param in vgg19.features[i].parameters():
            param.requires_grad = False
elif freeze_level == 4:
    for i in range(10):
        for param in vgg19.features[i].parameters():
            param.requires_grad = False
elif freeze_level == 5:
    for i in range(5):
        for param in vgg19.features[i].parameters():
            param.requires_grad = False
elif freeze_level == 6:
    pass
else:
    assert True, "Freeze level is out of range!!"

vgg19_place365 = vgg19.to(device)


# 4. Record setting
ckpt_dir = './data/fc_option{}_trainable_option{}_ckpt_{}_{}_{}'.format(FC_option, freeze_level, class_num, data_per_class_num, batch_size)
assert not os.path.isdir(ckpt_dir), 'remove the ckpt folder'
os.mkdir(ckpt_dir)
f = open(ckpt_dir + '/log.txt', 'w')

writer1 = SummaryWriter(comment="train")
writer2 = SummaryWriter(comment="valid")


# 5. Training setting   ** option
'''
1. epochs - fixed
 : default = 100
How many times to run training

2. loss_func - fixed
 : default = nn.CrossEntropyLoss()
Use CrossEntropy loss whenever possible

3. optimizer - fixed
 : default = Adam
Use Adam whenever possible

4. initial_lr - fixed
 : default = 1e-2
initial learning rate. 

5. exp_lr_scheduler - fixed
 : default = StepLR, step_size=10, gamma=0.1
Experiment with different scheduler options and choose one. (MultiplicativeLR, StepLR, ExponentialLR, CosineAnnealingLR)
'''


epochs = 100
loss_func = nn.CrossEntropyLoss()
initial_lr = 1e-2
optimizer = optim.Adam(vgg19_place365.parameters(), lr=initial_lr)
#exp_lr_scheduler = scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)  # https://pytorch.org/docs/stable/optim.html
exp_lr_scheduler = scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

ckpt_interval = int(iter_num / 2)


# 6. Training   ** option
'''
1. train_with_ckpt
 : default = False
if you train from checkpoint, change this to True

2. epoch_start
 : default = 0
if you train from checkpoint, change this to epoch that you want to start

3. ckpt_file_name
 : default = "ckpt_epoch_x_batch_id_y.pth" 
if you train from checkpoint, change this depending on your ckpt file name

'''

train_with_ckpt = False
epoch_start = 0
ckpt_file_name = "ckpt_epoch_x_batch_id_y.pth"

if train_with_ckpt:
    checkpoint = torch.load(os.path.join(ckpt_dir, ckpt_file_name))
    vgg19_place365.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']

for epoch in range(epoch_start, epochs):
    f.write("Epoch: {}/{}\n".format(epoch + 1, epochs))
    epoch_start = time.time()
    vgg19_place365.train()
    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_data)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        #for param in vgg19_place365.features[7].parameters():
        #    print(param)

        optimizer.zero_grad()
        outputs = vgg19_place365(inputs)
        loss = loss_func(outputs, labels)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        # train accuracy
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))

        train_loss += loss.item() * inputs.size(0)
        train_acc += acc.item() * inputs.size(0)

        # print(loss)

        # record log
        writer1.add_scalar("Batch/Train-loss", loss, (i + 1) + int(train_size / batch_size) * epoch)
        writer1.add_scalar("Batch/Train-accuracy", acc, (i + 1) + int(train_size / batch_size) * epoch)

        if ckpt_dir is not None and (i + 1) % ckpt_interval == 0:
            ckpt_filename = 'ckpt_epoch_' + str(epoch + 1) + '_batch_id_' + str(i + 1) + '.pth'
            f.write(str(epoch + 1) + ' checkpoint, ' + str(i + 1) + ' th batch is saved!\n')
            ckpt_model_path = os.path.join(ckpt_dir, ckpt_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': vgg19_place365.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, ckpt_model_path)
    print("Epoch: {}/{}".format(epoch + 1, epochs))

    # scheduler check
    exp_lr_scheduler.step()
    f.write("learning rate: " + str(optimizer.param_groups[0]['lr']) + "\n")

    # validation (check over-fitting & under-fitting looking at the tensorboard's loss & accuracy curve)
    with torch.no_grad():
        vgg19_place365.eval()
        for j, (inputs, labels) in enumerate(valid_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = vgg19_place365(inputs)
            loss = loss_func(outputs, labels)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            valid_loss += loss.item() * inputs.size(0)
            valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_size
        avg_train_acc = train_acc / float(train_size)
        avg_valid_loss = valid_loss / valid_size
        avg_valid_acc = valid_acc / float(valid_size)
        epoch_end = time.time()

        f.write("Epoch {} end - time: {}\n".format(epoch, epoch_end - epoch_start))
        f.write("Training Average - Loss: {}, Accuracy: {}\n".format(avg_train_loss, avg_train_acc * 100))
        f.write("Validation Average - Loss: {}, Accuracy: {}\n\n\n".format(avg_valid_loss, avg_valid_acc * 100))

        writer1.add_scalar("Epoch/avg_loss", avg_train_loss, epoch + 1)
        writer2.add_scalar("Epoch/avg_loss", avg_valid_loss, epoch + 1)
        writer1.add_scalar("Epoch/avg_accuracy", avg_train_acc, epoch + 1)
        writer2.add_scalar("Epoch/avg_accuracy", avg_valid_acc, epoch + 1)

writer1.close()
writer2.close()
f.close()
