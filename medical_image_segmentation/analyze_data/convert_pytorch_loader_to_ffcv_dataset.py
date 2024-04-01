from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
import os
import argparse
import torchvision
from medical_image_segmentation.analyze_data.pytorch_datasets import ChestXRayDataset


def parse_args():
    """Create args for command line interface."""
    parser = argparse.ArgumentParser(description="Process DICOM images and write them as a ffcv dataset.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use.", required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Where to write the train and test .beton files.",
        required=True,
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        help="Max resolution of side of an image.",
        default=224,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="number of workers to load/write images with.",
        default=int(os.environ.get("SLURM_CPUS_ON_NODE", "1")),
    )
    parser.add_argument("--test_only", action="store_true", help="Only writes test dataset.")
    parser.add_argument("--train_only", action="store_true", help="Only writes train dataset")

    return parser.parse_args()


def get_dataset(dataset_name, num_workers):
    """Dispatches the dataset name to the correct train and test dataset pairs"""
    dataset_map = {
        "cifar10": get_cifar10_datasets,
        "cifar100": get_cifar100_datasets,
        "imagenet": get_imagenet_datasets,
        "nih_chest_x_ray": get_nih_x_ray_datasets,
    }
    return dataset_map[dataset_name](num_workers)


def get_cifar10_datasets(num_workers=1):
    trainset = torchvision.datasets.CIFAR10(root="/tmp/cifar10", train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root="/tmp/cifar10", train=False, download=True)

    return trainset, testset


def get_cifar100_datasets(num_workers=1):
    trainset = torchvision.datasets.CIFAR100(root="/tmp/cifar100", train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root="/tmp/cifar100", train=False, download=True)

    return trainset, testset


def get_nih_x_ray_datasets(num_workers=1):
    csv_path = "/scratch/gpfs/eh0560/repos/medical-image-segmentation/data/nih_chest_x_ray_subset_info/original_image_path_to_label.csv"
    testset = ChestXRayDataset(csv_path)

    return None, testset


def get_imagenet_datasets(num_workers=1):
    trainset = torchvision.datasets.ImageNet(
        "/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization/",
        split="train",
    )
    testset = torchvision.datasets.ImageNet(
        "/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization/",
        split="val",
    )

    return trainset, testset


def create_writer(output_path, max_resolution):
    writer = DatasetWriter(
        output_path,
        {"image": RGBImageField(max_resolution=max_resolution), "label": IntField()},
    )
    return writer


def main():
    args = parse_args()
    if args.test_only and args.train_only:
        raise ValueError("Cannot have both train_only and test_only")

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset, test_dataset = get_dataset(args.dataset_name, args.num_workers)

    if not args.test_only:
        train_output_path = os.path.join(args.output_dir, f"{args.dataset_name}_{args.max_resolution}_train.beton")
        train_writer = create_writer(train_output_path, args.max_resolution)
        train_writer.from_indexed_dataset(train_dataset)

    if not args.train_only:
        test_output_path = os.path.join(args.output_dir, f"{args.dataset_name}_{args.max_resolution}_test.beton")
        test_writer = create_writer(test_output_path, args.max_resolution)
        test_writer.from_indexed_dataset(test_dataset)


if __name__ == "__main__":
    main()
