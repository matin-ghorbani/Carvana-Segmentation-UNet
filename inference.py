from argparse import ArgumentParser, BooleanOptionalAction

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt

from models import UNet
import utils as ul
import config


def load_model(checkpoint_path: str, in_channels: int = 3, out_channels: int = 1) -> UNet:
    model = UNet(in_channels=in_channels,
                 out_channels=out_channels).to(config.DEVICE)
    ul.load_checkpoint(torch.load(checkpoint_path), model)
    model.eval()
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(config.DEVICE)


def predict(model: UNet, image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        preds = torch.sigmoid(model(image))
        preds = (preds > .5).float()

    return preds


def overlay_prediction(image_path: str, prediction: torch.Tensor, save: bool = True) -> None:
    original_image = Image.open(image_path).convert('RGB')
    prediction = prediction.squeeze().cpu().numpy()

    # Create an empty mask
    mask = Image.new('RGBA', original_image.size)

    # Resize the prediction to match the original image size
    result_img = Image.fromarray((prediction * 255).astype(np.uint8))
    result_img = ImageOps.fit(
        result_img, original_image.size, method=Image.NEAREST)
    result_img = result_img.convert('L')

    # Create a polygon mask with transparency
    draw = ImageDraw.Draw(mask)
    for y in range(result_img.height):
        for x in range(result_img.width):
            if result_img.getpixel((x, y)) > 0:
                # Red color with 50% transparency
                draw.point((x, y), fill=(255, 0, 0, 128))

    # Overlay the mask on the original image
    overlayed_image = Image.alpha_composite(
        original_image.convert('RGBA'), mask)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(overlayed_image)
    plt.axis('off')
    plt.show()

    if save:
        overlayed_image.save('result.png', format='PNG')


def main(image_path: str, checkpoint_path: str, save: bool = True):
    model = load_model(checkpoint_path)
    image_tensor = preprocess_image(image_path)
    prediction = predict(model, image_tensor)
    overlay_prediction(image_path, prediction, save)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='The model path', required=True)
    parser.add_argument('--img', type=str,
                        help='The img path', required=True)
    parser.add_argument('--save', type=bool, action=BooleanOptionalAction,
                        help='Wether save the predication or not', default=True)

    args = parser.parse_args()

    main(args.img, args.model, args.save)
