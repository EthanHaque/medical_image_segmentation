from typing import Tuple
import torchvision
import pytorch_lightning as pl
from medical_image_segmentation.train.train_ssl import SelfSupervisedLearner
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch
import numpy as np
from medical_image_segmentation.train.data_loaders.ffcv_loader import create_train_loader_ssl, create_val_loader_ssl
from tqdm import tqdm


@torch.no_grad()
def compute_embeddings(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the embeddings output by a model on a particular dataset.

    Parameters
    ----------
    model: torch.nn.Module The model to use to the compute the embeddings.
    loader: torch.utils.data.DataLoader The DataLoader that applies any transformations to the input images and loads the images.
    device: torch.device The device to send the data to.

    Returns
    -------

    """
    embeddings = []
    ground_truth = []
    for batch in tqdm(loader, desc="Computing embeddings"):
        images = batch[0]
        labels = batch[1]

        x = images.to(device)
        target = labels.to(device)

        feature = model.forward(x, return_embedding=True)[0].flatten(start_dim=1)
        feature = F.normalize(feature, dim=1)

        embeddings.append(feature)
        ground_truth.append(target)

    output_embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    output_labels = torch.cat(ground_truth, dim=0).cpu().numpy()

    return output_embeddings, output_labels


def main():
    resnet = torchvision.models.resnet18(weights=None)

    model = SelfSupervisedLearner.load_from_checkpoint(r"logs/lightning_logs/version_71/checkpoints/epoch=51-step=16224.ckpt",
                                                       net=resnet,
                                                       image_size=56,
                                                       hidden_layer="avgpool",
                                                       projection_size=256,
                                                       projection_hidden_size=4096,
                                                       moving_average_decay=0.99,
                                                       )

    model.eval()

    train_loader = create_train_loader_ssl(
        this_device=torch.device("cuda:0"),
        beton_file_path="/scratch/gpfs/eh0560/data/imagenet_ffcv/imagenet_train.beton",
        batch_size=128,
        num_workers=8,
        num_gpus=1,
        image_size=56,
        in_memory=False,
        subset_size=1024,
    )

    val_loader = create_val_loader_ssl(
        this_device=torch.device("cuda:0"),
        beton_file_path="/scratch/gpfs/eh0560/data/imagenet_ffcv/imagenet_val.beton",
        batch_size=128,
        num_workers=8,
        num_gpus=1,
        image_size=56,
        in_memory=False,
        subset_size=128,
    )

    train_embeddings, train_ground_truth = compute_embeddings(model, train_loader, torch.device("cuda:0"))
    val_embeddings, val_ground_truth = compute_embeddings(model, val_loader, torch.device("cuda:0"))

    print(train_embeddings.shape, val_embeddings.shape)
    print(train_ground_truth.shape, val_ground_truth.shape)



if __name__ == '__main__':
    main()
