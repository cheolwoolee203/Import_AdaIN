from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


content = ""  # './input/content/texturecontent0.jpg'
content_dir = './input/content_test'
style = ""  # './input/style/00000161.jpg'
style_dir = './input/style_test'
vgg_dir = './input/models/vgg_normalised.pth'
decoder_dir = './input/models/decoder_iter_10000.pth.tar'
content_size = 512
style_size = 512
crop = 'store_true'
save_ext = '.jpg'
output = './output'
alpha = 1.0
preserve_color ='store_true'
style_interpolation_weights = ''

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Either --content or --contentDir should be given.
content_paths = []
assert (content or content_dir)
if content:
    content_paths = [Path(content)]
else:
    content_dir_Path = Path(content_dir)
    content_paths_folder = [f for f in content_dir_Path.glob('*')]
    for folder_name in content_paths_folder:
        folder_name_Path = Path(folder_name)
        content_paths += [f for f in folder_name_Path.glob('*')]

# Either --style or --styleDir should be given.
style_paths = []
assert (style or style_dir)
if style:
    style_paths = style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(style)]
    else:
        do_interpolation = True
        assert (style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir_Path = Path(style_dir)
    style_paths_folder = [f for f in style_dir_Path.glob('*')]
    for folder_name in style_paths_folder:
        folder_name_Path = Path(folder_name)
        style_paths += [f for f in folder_name_Path.glob('*')]

output_dir = Path(output)
output_dir.mkdir(exist_ok=True, parents=True)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(decoder_dir))
vgg.load_state_dict(torch.load(vgg_dir))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)

for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    alpha, interpolation_weights)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, save_ext)
        save_image(output, str(output_name))

    else:  # process one content and one style
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            if preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        alpha)
            output = output.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, save_ext)
            save_image(output, str(output_name))

