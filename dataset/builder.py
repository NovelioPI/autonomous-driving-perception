from pathlib import Path
import pickle
from torch.utils import data
import logging

def build_dataset_train(
    dataset: str,
    input_size: tuple,
    batch_size: int,
    num_workers: int,
    random_scale: bool,
    random_mirror: bool,
    use_val: bool = True,
    root: str = './dataset/'
):
    """
    Build the training dataset based on the specified dataset type.

    Returns:
        tuple: Dataset statistics, DataLoader for training, DataLoader for validation (or None).
    """
    root_path = Path(root)
    data_dir = root_path / dataset
    inform_dir = Path('./dataset/inform') 
    inform_data_file = inform_dir / f"{dataset}_inform.pkl"

    # Check if inform_data_file exists; create statistics if not
    if not inform_data_file.is_file():
        print(f"Statistics file not found: {inform_data_file}")
        inform_dir.mkdir(parents=True, exist_ok=True)
        if dataset == "cityscapes":
            from dataset.cityscapes import CityscapesInform
            data_inform = CityscapesInform(data_dir)
        else:
            raise NotImplementedError(f"Dataset '{dataset}' not supported.")
        data_stat = data_inform.read_and_save_statistics(inform_data_file)
    else:
        print(f"Found statistics file: {inform_data_file}")
        with open(inform_data_file, "rb") as f:
            data_stat = pickle.load(f)

    if dataset == "cityscapes":
        from dataset.cityscapes import CityscapesDataset
        def get_loader(split, batch, scale, mirror, shuffle, drop_last):
            return data.DataLoader(
                CityscapesDataset(root, split=split, stat=data_stat,
                                  crop_size=input_size, scale=scale, mirror=mirror),
                batch_size=batch, shuffle=shuffle, num_workers=num_workers,
                pin_memory=True, drop_last=drop_last
            )
        train_loader = get_loader('train', batch_size, random_scale, random_mirror, True, True)
        val_loader = get_loader('val', 1, False, False, False, True) if use_val else None
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported.")

    return data_stat, train_loader, val_loader
