from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, Squeeze, ToDevice, RandomHorizontalFlip, ImageMixup, Rotate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_train_loader_ssl(this_device: str, beton_file_path: str, batch_size: int, num_workers: int,
                        distributed: bool, in_memory: bool) -> Loader:
    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    pipeline_1 = [
        NDArrayDecoder(),
    ]
    pipeline_2 = [
        NDArrayDecoder(),
    ]
    pipelines = {
        "image": pipeline_1,
        "image_0": pipeline_2,
    }

    custom_field_mapper = {"image_0": "image"}

    train_loader = Loader(beton_file_path,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          order=order,
                          os_cache=in_memory,
                          drop_last=True,
                          pipelines=pipelines,
                          distributed=distributed,
                          custom_field_mapper=custom_field_mapper
                          )

    return train_loader


if __name__ == "__main__":
    this_device = f"cuda:0"
    beton_file_path = "/scratch/gpfs/RUSTOW/med_datasets/ffcv_datasets/radiology_1M.beton"
    batch_size = 256
    num_workers = 8
    distributed = False
    in_memory = True

    loader = create_train_loader_ssl(this_device, beton_file_path, batch_size, num_workers, distributed, in_memory)

    iterator = tqdm(loader)
    for loaders in iterator:
        view_1 = loaders[0]
        view_2 = loaders[1]
        concat_image = np.concatenate([view_1, view_2], axis=2)

        # for pair in range(concat_image.shape[0]):
        #     image_arr = concat_image[pair]
        #     plt.figure(figsize=(10, 6))
        #     plt.imshow(image_arr, cmap="binary")
        #     plt.show()
