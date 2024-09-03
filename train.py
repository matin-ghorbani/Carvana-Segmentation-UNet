import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from models import UNet
import utils as ul
import config


def train_fn(
        loader: DataLoader,
        model: UNet,
        optimizer: optim.Adam,
        loss_fn: nn.BCEWithLogitsLoss,
        scaler: GradScaler
):
    data: torch.Tensor
    targets: torch.Tensor

    loop = tqdm(loader)
    for idx, (data, targets) in enumerate(loop):
        data = data.to(config.DEVICE)
        targets = targets.float().unsqueeze(1).to(config.DEVICE)

        with autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main() -> None:
    train_transforms = A.Compose([
        A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.),
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.1),
        A.Normalize(
            mean=[0., 0., 0.],
            std=[1., 1., 1.],
            max_pixel_value=255.
        ),
        ToTensorV2()
    ])
    valid_transforms = A.Compose([
        A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
        A.Normalize(
            mean=[0., 0., 0.],
            std=[1., 1., 1.],
            max_pixel_value=255.
        ),
        ToTensorV2()
    ])

    # If we have multiple classes we have to change out_channels to the number of classes
    # And change the loss function to CrossEntropyLoss
    model = UNet(in_channels=3, out_channels=1).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader, valid_loader = ul.get_loaders(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        config.VALID_IMG_DIR,
        config.VALID_MASK_DIR,
        config.BATCH_SIZE,
        train_transforms,
        valid_transforms,
        config.NUM_WORKERS,
        config.PIN_MEMORY
    )

    if config.LOAD_MODEL:
        ul.load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)

    scaler = GradScaler()
    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler
        )

        # Save Model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        ul.save_checkpoint(checkpoint)

        # Check Accuracy
        ul.check_accuracy(valid_loader, model, device=config.DEVICE)

        # Save Some Examples
        ul.save_predictions_as_imgs(valid_loader, model, folder='saved_images', device=config.DEVICE)


if __name__ == '__main__':
    main()
