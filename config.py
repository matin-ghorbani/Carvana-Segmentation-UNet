from torch.cuda import is_available

LEARNING_RATE: float = 1e-4
DEVICE: str = 'cuda' if is_available() else 'cpu'
BATCH_SIZE:int = 16
NUM_EPOCHS:int = 3
NUM_WORKERS:int = 2
IMAGE_HEIGHT:int = 160  # 1280 originally
IMAGE_WIDTH:int = 240  # 1918 originally
PIN_MEMORY: bool = True
LOAD_MODEL: bool = False  # Make it true if you want to use a pretrained model
TRAIN_IMG_DIR: str = 'data/train_images/'
TRAIN_MASK_DIR: str = 'data/train_masks/'
VALID_IMG_DIR: str = 'data/val_images/'
VALID_MASK_DIR: str = 'data/val_masks/'
