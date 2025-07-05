from pathlib import Path
from torch.utils import data
import numpy as np
import pickle
import cv2
from tqdm import tqdm
import logging


class CityscapesDataset(data.Dataset):
    def __init__(
        self, root: Path, split: str = 'train', stat: dict = None, crop_size: tuple = (512, 1024),
        scale: bool = False, mirror: bool = False
    ):
        self.root = Path(root)
        self.split = split

        img_dir = self.root / 'cityscapes' / 'leftImg8bit' / split
        label_dir = self.root / 'cityscapes' / 'gtFine' / split

        self.images = sorted(img_dir.glob('**/*.png'))
        self.labels = sorted(label_dir.glob('**/*_labelTrainIds.png'))

        if not self.images:
            raise FileNotFoundError(f"No images found in {img_dir}")
        if not self.labels:
            raise FileNotFoundError(f"No labels found in {label_dir}")
        if len(self.images) != len(self.labels):
            raise ValueError(f"Number of images ({len(self.images)}) and labels ({len(self.labels)}) do not match.")

        # Load statistics if provided
        self.mean = stat['mean'] if stat is not None else np.array([128.0, 128.0, 128.0], dtype=np.float32)
        self.std = stat['std'] if stat is not None else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.num_classes = stat['num_classes'] if stat is not None else 19
        self.ignore_index = stat['ignore_index'] if stat is not None else 255

        # Augmentation parameters
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mirror = mirror

        print(f"{split} dataset contains {len(self.images)} images and {len(self.labels)} labels.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        img = cv2.imread(str(img_path))
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

        if img is None or label is None:
            raise FileNotFoundError(f"Image or label not found: {img_path}, {label_path}")

        size = img.shape
        name = img_path.stem

        # Resize image and label if scale is enabled
        if self.scale:
            f_scale = np.random.uniform(0.75, 2.0)
            img = cv2.resize(img, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32)

        # Pad if needed (still HWC)
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_index)

        # Crop
        img_h, img_w = label.shape  # Update after padding
        h_off = np.random.randint(0, max(img_h - self.crop_h + 1, 1))
        w_off = np.random.randint(0, max(img_w - self.crop_w + 1, 1))
        img = img[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        label = label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]

        # Augmentation: Random horizontal flip
        if self.mirror and np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            label = np.fliplr(label).copy()

        # Normalize and channel conversion (BGR -> RGB), then transpose to CHW
        img = (img - self.mean) / self.std
        img = img[:, :, ::-1]
        img = img.transpose(2, 0, 1)  # HWC to CHW

        return img.copy(), label.copy(), np.array(size), name



class CityscapesInform(data.Dataset):
    """
    Statistical information for Cityscapes train set such as mean, std, class distribution.
    The class is employed to tackle class imbalance.
    Computes simple average of means and stds across images.
    """
    def __init__(self, root: Path, norm_val: float = 1.10):
        self.root = Path(root)
        self.images = sorted((self.root / 'cityscapes' / 'leftImg8bit' / 'train').glob('**/*.png'))
        self.labels = sorted((self.root / 'cityscapes' / 'gtFine' / 'train').glob('**/*_labelTrainIds.png'))
        
        if len(self.images) != len(self.labels):
            raise ValueError("Number of images and labels do not match!")
        
        self.num_classes = 19
        self.ignore_index = 255
        self.norm_val = norm_val
        self.class_weights = np.ones(self.num_classes, dtype=np.float32)
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        logging.basicConfig(level=logging.INFO)

    def _read_data(self) -> None:
        global_hist = np.zeros(self.num_classes, dtype=np.float64)
        valid_file_count = 0
        
        for img_path, label_path in tqdm(zip(self.images, self.labels), total=len(self.images), desc="Reading images and labels"):
            img = cv2.imread(str(img_path))
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None or label is None:
                logging.warning(f"Missing file: {img_path} or {label_path}")
                continue
            
            # Update global histogram (excluding ignore_index)
            mask = (label != self.ignore_index)
            unique, counts = np.unique(label[mask], return_counts=True)
            global_hist[unique] += counts
            
            # Update mean and std
            self.mean += np.mean(img, axis=(0, 1))
            self.std += np.std(img, axis=(0, 1))
            valid_file_count += 1
            
            # Check for unknown classes
            invalid = unique[(unique >= self.num_classes) | (unique < 0)]
            if invalid.size > 0:
                logging.warning(f"Unknown class(es) in label: {label_path} - {invalid}")
            
        if valid_file_count == 0:
            raise RuntimeError("No valid images found.")

        self.mean /= valid_file_count
        self.std /= valid_file_count
        
        # Calculate class weights (avoid zero division)
        zero_mask = global_hist == 0
        if np.any(zero_mask):
            logging.warning("Some classes have zero samples, setting their weights to 1.")
            global_hist[zero_mask] = 1
        norm_hist = global_hist / global_hist.sum()
        self.class_weights = 1.0 / np.log(self.norm_val + norm_hist)

    def read_and_save_statistics(self, output_file: str, rewrite: bool = False):
        output_file = Path(output_file)
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Reading data and calculating statistics...")
        
        if output_file.exists() and not rewrite:
            logging.info(f"Statistics file {output_file} already exists. Loading existing statistics.")
            with open(output_file, 'rb') as f:
                return pickle.load(f)
            
        self._read_data()
        statistics = {
            'mean': self.mean,
            'std': self.std,
            'class_weights': self.class_weights,
            'num_classes': self.num_classes,
            'ignore_index': self.ignore_index
        }
        with open(output_file, 'wb') as f:
            pickle.dump(statistics, f)
        logging.info(f"Statistics saved to {output_file}")
        return statistics


if __name__ == "__main__":
    # Example usage
    root = '/media/esr/ssd0'
    inform = CityscapesInform(root)
    stats = inform.read_and_save_statistics('./dataset/inform/cityscapes_inform.pkl')
    print("Statistics:", stats)
    
    dataset = CityscapesDataset(root, split='train', stat=stats, crop_size=(512, 1024), scale=True, mirror=True)
    print(f"Dataset size: {len(dataset)}")
    img, label, size, name = dataset[0]
    print(f"Image shape: {img.shape}, Label shape: {label.shape}, Size: {size}, Name: {name}")