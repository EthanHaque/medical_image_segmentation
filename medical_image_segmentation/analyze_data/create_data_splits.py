import os
import json
import random
import argparse


def create_split(image_ids, train_size=0.7, val_size=0.2, test_size=0.1):
    if not (0.999 <= train_size + val_size + test_size <= 1.001):
        raise ValueError("The sum of train_size, val_size, and test_size must be within 0.001 of 1.0")

    random.shuffle(image_ids)

    number_images = len(image_ids)
    number_train = int(number_images * train_size)
    number_val = int(number_images * val_size)

    splits = {
        "train": image_ids[:number_train],
        "val": image_ids[number_train:number_train + number_val],
        "test": image_ids[number_train + number_val:]
    }

    return splits


def create_split_by_percent(image_ids, train_size=0.7, val_size=0.2, test_size=0.1,
                            train_percents=(0.1, 0.25, 0.50, 1.0)):
    splits = {}
    base_split = create_split(image_ids, train_size, val_size, test_size)
    for percent in train_percents:
        subset = base_split.copy()
        len_train = len(subset["train"])
        new_samples = int(len_train * percent)
        random.shuffle(subset["train"])
        subset["train"] = subset["train"][:new_samples]
        splits[percent] = subset

    return splits


def get_ids(image_dir):
    files = os.listdir(image_dir)
    ids = []
    for file in files:
        image_id = file.split("_")[-1].split(".")[0]
        image_id = int(image_id)
        ids.append(image_id)
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, help="Root dir with image files whose names have unique ids.")
    parser.add_argument("--output_dir", type=str, help="Where to write output json files with splits.")

    return parser.parse_args()


def main():
    args = parse_args()
    ids = get_ids(args.image_dir)
    splits = create_split_by_percent(ids)
    for size, split in splits.items():
        percent_of_train = int(size * 100)
        print(f"{percent_of_train}", json.dumps(split))


if __name__ == "__main__":
    main()
