import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_training_dataloader(root_path, batch_size=16, num_workers=2, shuffle=True, size=(128, 128)):
    """ return training dataloader
    Args:
        root_path: path to training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: training_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        torchvision.transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    training_dataset = torchvision.datasets.ImageFolder(
        root=root_path, transform=transform_train)
    print(training_dataset.class_to_idx)
    training_loader = DataLoader(
        training_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return training_loader

def get_valid_dataloader(root_path, batch_size=16, num_workers=2, shuffle=False, size=(128, 128)):
    """ return validation dataloader
    Args:
        root_path: path to validation dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: valid_loader:torch dataloader object
    """

    transform_valid = transforms.Compose([
        torchvision.transforms.Resize(size),
        transforms.ToTensor()
    ])

    valid_dataset = torchvision.datasets.ImageFolder(
        root=root_path, transform=transform_valid)
    print(valid_dataset.class_to_idx)
    valid_loader = DataLoader(
        valid_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return valid_loader



def get_test_dataloader(root_path, batch_size=16, num_workers=2, shuffle=False, size=(128, 128)):

    test_transform = transforms.Compose([
        torchvision.transforms.Resize(size),
        transforms.ToTensor(),
    ])

    test_dataset = torchvision.datasets.ImageFolder(
        root_path, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


    return test_loader
