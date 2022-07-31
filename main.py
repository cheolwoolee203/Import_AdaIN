from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
# from torchinfo import summary
from tqdm import tqdm
import os
import random

import net
from sampler import InfiniteSamplerWrapper


def train_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform, style=False, train=False, places_num=274):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        if not style:
            self.paths = list(Path(self.root).glob('*'))
        elif style:

            if train:
                self.paths = []
                filelist = list(os.listdir(self.root))
                for filename in filelist:
                    img_list = list(Path(self.root + '/' + filename).glob('*'))
                    img_list_r = random.choices(img_list, k=places_num)
                    self.paths += img_list_r
                self.paths = self.paths[:-10]

            elif not train:
                self.paths = []
                filelist = list(os.listdir(self.root))
                for filename in filelist:
                    img_list = list(Path(self.root + '/' + filename).glob('*'))
                    img_list_r = random.choices(img_list, k=44)
                    self.paths += img_list_r
                self.paths = self.paths[:-60]

        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 1e-4
    lr_decay = 5e-5
    lr_b = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_b


def main():
    cudnn.benchmark = True
    Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
    # Disable OSError: image file is truncated
    ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---hyper Parameter---------------------------------------------
    optimizer_what = 'Adam'     # Adam, NAdam, SGD
    batch_size = 16             # 16, 8, 4
    pooling = 'Max'             # Max, Avg
    feature_layer = 4           # relu4-1 = 4, relu3-1 = 3, relu2-1 = 2
    data_size = '160k'          # 160k, 40k, 5k, Art40k
    alpha_beta = 10.0           # 10.0 (S = 10.0, C = 1.0),
                                # 1.0 (S = 5.0, C = 5.0),
                                # 0.1 (S = 1.0, C = 10.0)
# ----------------------------------------------------------------

    content_dir = './input/train2017/Train'
    content_val_dir = './input/train2017/Val'
    style_dir = './input/archive/train'
    style_val_dir = './input/archive/val'
    vgg_dir = './input/models/vgg_normalised.pth'
    save_dir = './input/train'
    log_dir = './input/train'
    epoch = 50
    lr = 1e-4
    lr_decay = 5e-5
    n_threads = 4
    n_threads_val = 0
    save_model_interval = 10000
    max_iter = 6250
    max_iter_val = 1000
    places_num = 274
    style_weight = 1.0
    content_weight = 1.0

    if data_size == '160k':
        max_iter = 6250
        places_num = 274
    elif data_size == '40k':
        max_iter = 2500
        places_num = 110
    elif data_size == '5k':
        max_iter = 313
        places_num = 14
    elif data_size == 'Ark40k':
        max_iter = 2500
        content_dir = './input/train2017/Train'
        content_val_dir = './input/train2017/Val'

    if alpha_beta == 10.0:
        style_weight = 10.0
        content_weight = 1.0
    elif alpha_beta == 1.0:
        style_weight = 5.0
        content_weight = 5.0
    elif alpha_beta == 0.1:
        style_weight = 1.0
        content_weight = 10.0

    if batch_size == 16:
        max_iter_val = 1000
    elif batch_size == 8:
        max_iter_val = 2000
        max_iter *= 2
    elif batch_size == 4:
        max_iter_val = 4000
        max_iter *= 4

    device = torch.device('cuda')
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True, parents=True)
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True, parents=True)

    decoder = net.decoder
    vgg = net.vgg
    if pooling == 'Max':
        vgg = net.vgg
    elif pooling == 'Avg':
        vgg = net.vgg_Avg

    vgg.load_state_dict(torch.load(vgg_dir))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    network = net.Net(vgg, decoder)
    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()
    content_val_tf = train_transform()
    style_val_tf = train_transform()

    content_dataset = FlatFolderDataset(content_dir, content_tf, False, False, places_num)
    style_dataset = FlatFolderDataset(style_dir, style_tf, True, True, places_num)
    content_val_dataset = FlatFolderDataset(content_val_dir, content_val_tf)
    style_val_dataset = FlatFolderDataset(style_val_dir, style_val_tf, True)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=n_threads))

    content_val_iter = iter(data.DataLoader(
        content_val_dataset, batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(content_val_dataset),
        num_workers=n_threads_val))
    style_val_iter = iter(data.DataLoader(
        style_val_dataset, batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(style_val_dataset),
        num_workers=n_threads_val))

    # summary(network, input_size=(batch_size, 3, 224, 224))

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=lr)
    if optimizer_what == 'Adam':
        optimizer = torch.optim.Adam(network.decoder.parameters(), lr=lr)
    elif optimizer_what == 'NAdam':
        optimizer = torch.optim.NAdam(network.decoder.parameters(), lr=lr)
    elif optimizer_what == 'SGD':
        optimizer = torch.optim.SGD(network.decoder.parameters(), lr=lr)

    for k in range(epoch):
        comment = f' epoch = {k}'
        writer = SummaryWriter(comment=comment)

        for i in tqdm(range(max_iter)):
            adjust_learning_rate(optimizer, iteration_count=k)
            content_images = next(content_iter).to(device)
            style_images = next(style_iter).to(device)
            loss_c, loss_s = network(content_images, style_images, feature_layer)
            loss_c = content_weight * loss_c
            loss_s = style_weight * loss_s
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss_content', loss_c.item(), i + 1)
            writer.add_scalar('loss_style', loss_s.item(), i + 1)

            if (i + 1) % save_model_interval == 0 or (i + 1) == max_iter:
                state_dict = net.decoder.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict, save_dir + '/' + 'decoder_iter_{:d}_epoch_{:d}.pth.tar'.format((i + 1), (k + 1)))

        # for j in tqdm(range(max_iter_val)):
        #     content_val_images = next(content_val_iter).to(device)
        #     style_val_images = next(style_val_iter).to(device)
        #     loss_c_val, loss_s_val = network(content_val_images, style_val_images)
        #     loss_c_val = content_weight * loss_c_val
        #     loss_s_val = style_weight * loss_s_val
        #
        #     writer.add_scalar('loss_content_val', loss_c_val.item(), j + 1)
        #     writer.add_scalar('loss_style_val', loss_s_val.item(), j + 1)

        writer.close()


if __name__ == '__main__':
    main()

