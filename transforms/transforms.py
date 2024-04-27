from torchvision import transforms
from .my import MyNormalize, MyToTensor, MyRandomHorizontalFlip, MyRandomVerticalFlip, MyRandomRotation, MyResize


def train_transforms(datasets, input_size_h, input_size_w):
    return transforms.Compose([
        MyNormalize(datasets, train=True),
        MyToTensor(),
        MyRandomHorizontalFlip(p=0.5),
        MyRandomVerticalFlip(p=0.5),
        MyRandomRotation(p=0.5, degree=[0, 360]),
        MyResize(input_size_h, input_size_w)
    ])


def test_transformers(datasets, input_size_h, input_size_w):
    return transforms.Compose([
        MyNormalize(datasets, train=False),
        MyToTensor(),
        MyResize(input_size_h, input_size_w)
    ])