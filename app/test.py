import albumentations as A


def get_training_augmentation():
    train_transform = [
        A.RandomCrop(height=768, width=768, always_apply=True),

        A.VerticalFlip(),
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.Transpose(),
        A.IAAAdditiveGaussianNoise(),
        A.IAASharpen(),
        A.HueSaturationValue(),
        A.IAAPerspective(),
        A.IAAAffine(),
        A.RandomBrightnessContrast()
    ]
    return A.Compose(list=train_transform, p=0.5)
