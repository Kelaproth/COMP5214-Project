import torch
from utils import image_converter
from PIL import Image, ImageDraw, ImageFont
from train import MODEL_LIST
from model.ipt import quantize

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoint configuration
checkpoint = "./save/srresnet_checkpoint_99.pth.tar"
model_type = 'srresnet'

# Load models
def load_model(checkpoint_path, model_type, device):
    assert model_type in MODEL_LIST
    if model_type == 'srresnet':
        model = torch.load(checkpoint_path)['model'].to(device)
        model.eval()
    elif model_type == 'srgan':
        model = torch.load(checkpoint_path)['generator'].to(device)
        model.eval()
    elif model_type == 'vit':
        model = torch.load(checkpoint_path)['model'].to(device)
        model.eval()
    elif model_type == 'mae':
        model = torch.load(checkpoint_path)['model'].to(device)
        model.eval()
    # elif model_type == 'ipt':     
    #     checkpoint = Checkpoint(args)
    #     if checkpoint.ok:
    #         model = Model(args, checkpoint)
    #         if args.pretrain == '':
    #             args.pretrain = "./save/ipt/IPT_sr4.pt"
    #         state_dict = torch.load(args.pretrain)
    #         model.model.load_state_dict(state_dict, strict = False)
    #         model.to(device)
    #         model.eval()
    return model

def visualize_sampling(img, model_type, model, device, halve=False):
    """
    Visualizes the super-resolved images from the SRResNet and SRGAN for comparison with the bicubic-upsampled image
    and the original high-resolution (HR) image, as done in the paper.

    :param img: filepath of the HR iamge
    :param model: the model to be used
    :param halve: halve each dimension of the HR image to make sure it's not greater than the dimensions of your screen?
                  For instance, for a 2160p HR image, the LR image will be of 540p (1080p/4) resolution. On a 1080p screen,
                  you will therefore be looking at a comparison between a 540p LR image and a 1080p SR/HR image because
                  your 1080p screen can only display the 2160p SR/HR image at a downsampled 1080p. This is only an
                  APPARENT rescaling of 2x.
                  If you want to reduce HR resolution by a different extent, modify accordingly.
    """
    assert model_type in MODEL_LIST
    # Load image, downsample to obtain low-res version
    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert('RGB')
    if halve:
        hr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
                               Image.LANCZOS)
    lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
                           Image.BICUBIC)

    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    if model_type == 'srresnet' or model_type == 'srgan':
        # Super-resolution (SR) with SRResNet
        sr_img = model(image_converter(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
        sr_img = sr_img.squeeze(0).cpu().detach()
        sr_img = image_converter(sr_img, source='[-1, 1]', target='pil')
    elif model_type == 'vit':
        sr_img = model(image_converter(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
        sr_img = sr_img.squeeze(0).cpu().detach()
        sr_img = image_converter(sr_img, source='[-1, 1]', target='pil')
    elif model_type == 'ipt':
        sr_img = model(image_converter(lr_img, source='pil', target='[0, 255]').unsqueeze(0).to(device), 0)
        sr_img = quantize(sr_img)
        sr_img = sr_img.squeeze(0).cpu().detach().numpy()
        print(hr_img.height, hr_img.width)
        print(sr_img)
        print(sr_img.shape)
        sr_img = Image.fromarray(sr_img.astype('uint8').transpose(1, 2, 0), 'RGB')

    # Create grid
    margin = 40
    grid_img = Image.new('RGB', (3 * hr_img.width + 4 * margin, hr_img.height + 2 * margin), (255, 255, 255))

    # Font
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("calibril.ttf", size=23)
        # It will also look for this file in your OS's default fonts directory, where you may have the Calibri Light font installed if you have MS Office
        # Otherwise, use any TTF font of your choice
    except OSError:
        print(
            "Defaulting to a terrible font. To use a font of your choice, include the link to its TTF file in the function.")
        font = ImageFont.load_default()

    # Place bicubic-upsampled image
    grid_img.paste(bicubic_img, (margin, margin))
    text_size = font.getsize("Bicubic")
    draw.text(xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, margin - text_size[1] - 5], text="Bicubic",
              font=font,
              fill='black')

    # Place model sr image
    grid_img.paste(sr_img, (2 * margin + bicubic_img.width, margin))
    text_size = font.getsize("Model SR")
    draw.text(
        xy=[2 * margin + bicubic_img.width + sr_img.width / 2 - text_size[0] / 2, margin - text_size[1] - 5],
        text="Model SR", font=font, fill='black')

    # Place original HR image
    grid_img.paste(hr_img, (3 * margin + bicubic_img.width + sr_img.width, margin))
    text_size = font.getsize("Original HR")
    draw.text(xy=[3 * margin + bicubic_img.width + sr_img.width + hr_img.width / 2 - text_size[0] / 2,
                    margin - text_size[1] - 5], text="Original HR", font=font, fill='black')

    # Display grid
    grid_img.show()

    return grid_img

def visualize_original(lr_img_path, hr_img_path, model_type, model, halve=False):
    """
    Visualizes the super-resolved images from the SRResNet and SRGAN for comparison with the bicubic-upsampled image
    and the original high-resolution (HR) image, as done in the paper.

    :param img: filepath of the HR iamge
    :param model: the model to be used
    :param halve: halve each dimension of the HR image to make sure it's not greater than the dimensions of your screen?
                  For instance, for a 2160p HR image, the LR image will be of 540p (1080p/4) resolution. On a 1080p screen,
                  you will therefore be looking at a comparison between a 540p LR image and a 1080p SR/HR image because
                  your 1080p screen can only display the 2160p SR/HR image at a downsampled 1080p. This is only an
                  APPARENT rescaling of 2x.
                  If you want to reduce HR resolution by a different extent, modify accordingly.
    """
    assert model_type in MODEL_LIST
    # Load image, downsample to obtain low-res version
    lr_img = Image.open(lr_img_path, mode="r")
    lr_img = lr_img.convert('RGB')
    hr_img = Image.open(hr_img_path, mode="r")
    hr_img = hr_img.convert('RGB')
    if halve:
        hr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
                               Image.LANCZOS)
        lr_img = lr_img.resize((int(lr_img.width / 2), int(lr_img.height / 2)),
                               Image.LANCZOS)

    if model_type == 'srresnet' or model_type == 'srgan':
        # Super-resolution (SR) with SRResNet
        sr_img = model(image_converter(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
        sr_img = sr_img.squeeze(0).cpu().detach()
        sr_img = image_converter(sr_img, source='[-1, 1]', target='pil')
    elif model_type == 'vit':
        sr_img = model(image_converter(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
        sr_img = sr_img.squeeze(0).cpu().detach()
        sr_img = image_converter(sr_img, source='[-1, 1]', target='pil')
    elif model_type == 'ipt':
        sr_img = model(image_converter(lr_img, source='pil', target='[0, 255]').unsqueeze(0).to(device), 0)
        sr_img = quantize(sr_img)
        sr_img = sr_img.squeeze(0).cpu().detach()
        sr_img = image_converter(sr_img, source='[0, 255]', target='pil')


    # Create grid
    margin = 40
    grid_img = Image.new('RGB', (3 * hr_img.width + 4 * margin, hr_img.height + 2 * margin), (255, 255, 255))

    # Font
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("calibril.ttf", size=23)
        # It will also look for this file in your OS's default fonts directory, where you may have the Calibri Light font installed if you have MS Office
        # Otherwise, use any TTF font of your choice
    except OSError:
        print(
            "Defaulting to a terrible font. To use a font of your choice, include the link to its TTF file in the function.")
        font = ImageFont.load_default()

    # Place bicubic-upsampled image
    grid_img.paste(lr_img, (margin, margin))
    text_size = font.getsize("Bicubic")
    draw.text(xy=[margin + lr_img.width / 2 - text_size[0] / 2, margin - text_size[1] - 5], text="Bicubic",
              font=font,
              fill='black')

    # Place model sr image
    grid_img.paste(sr_img, (2 * margin + lr_img.width, margin))
    text_size = font.getsize("Model SR")
    draw.text(
        xy=[2 * margin + lr_img.width + sr_img.width / 2 - text_size[0] / 2, margin - text_size[1] - 5],
        text="Model SR", font=font, fill='black')

    # Place original HR image
    grid_img.paste(hr_img, (3 * margin + lr_img.width + sr_img.width, margin))
    text_size = font.getsize("Original HR")
    draw.text(xy=[3 * margin + lr_img.width + sr_img.width + hr_img.width / 2 - text_size[0] / 2,
                    margin - text_size[1] - 5], text="Original HR", font=font, fill='black')

    # Display grid
    grid_img.show()

    return grid_img


# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = load_model(checkpoint, model_type, device)
#     grid_img = visualize_sampling("./test/2.png", model_type, model, device)
