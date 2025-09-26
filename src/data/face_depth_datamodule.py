import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" 
import cv2
import json
###
import random
import open3d as o3d
###
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from typing import Optional, Dict, Any, Tuple, List, Union
import numpy as np
from PIL import Image
import random
import torchvision.transforms.functional as TF
# from src.utils.visualization import FaceDepthVisualizer


def check_existence(*file_paths):
    exist = True
    for file_path in file_paths:
        if not os.path.exists(file_path):
            exist = False
    return exist

def load_txt(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        K = []
        for line in lines:
            lsplit = line[:-1].split(" ")
            lsplit = np.asarray(lsplit).reshape(-1,3).astype(np.float32)
            K.append(lsplit)
    K = np.concatenate(K, axis=0)
    return K

def pad2square(*imgs):

    square_imgs = []
    for img in imgs:
        img_dim = np.shape(img)

        if len(img_dim)==3:
            img_h, img_w, img_c = img_dim

            pad_left = int((img_h-img_w)/2)
            pad_right = (img_h-img_w) - pad_left

            pad_left = np.zeros([img_h, pad_left, img_c], dtype=img.dtype)
            pad_right = np.zeros([img_h, pad_right, img_c], dtype=img.dtype)
            
        elif len(img_dim)== 2:
            img_h, img_w = img_dim
            pad_left = int((img_h-img_w)/2)
            pad_right = (img_h-img_w) - pad_left

            pad_left = np.zeros([img_h, pad_left], dtype=img.dtype)
            pad_right = np.zeros([img_h, pad_right], dtype=img.dtype)

        square_imgs.append(np.concatenate([pad_left, img, pad_right], axis=1))
    return square_imgs



def backproject_depth(K, depth_rn, img_cv):
    img = img_cv.astype(np.float32) / 255.0
    img_h, img_w = np.shape(depth_rn)[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h), indexing="xy")
    u = (u - cx) / cx
    v = (v - cy) / cy

    # xy1 = np.concatenate([u[:,:,None], v[:,:,None], np.ones_like(u[:,:,None])], axis=2) # HW3
    # d = depth_rn.reshape(img_h, img_w, 1)
    # xyz = xy1 * d
    valid = (depth_rn != 0).reshape(-1)

    

    xyz = np.concatenate([u[:,:,None], v[:,:,None], depth_rn[:,:,None]], axis=2) # HW3
    colors = img.reshape(-1, 3)
    colors = colors[:,::-1]
    xyz = xyz.reshape(-1, 3)

    xyz = xyz[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def normalize_depth(K, depth, nose_depth):
    f = K[0, 0]
    cx = K[0, 2]
    scale = f / cx    
   
    invalid = depth == 0
    depth = depth / nose_depth  # image place 1.0 +-
    depth = depth - 1
    depth *= scale        
    depth[invalid] = -1

    return depth

def apply_T(T, points):
    if len(T.shape) != 2 or len(points.shape) != 2:
        raise Exception("ERROR : the dimensions of transformation matrix and points are wrong.")

    points_ = np.matmul(T[:3, :3], points.transpose(1,0)).transpose(1,0) + T[:3, -1].reshape(-1,3)
    return points_

def make_origin(T_gk=np.eye(4), scale=1):
    points = np.array([[0,0,0],
                       [1,0,0],
                       [0,1,0],
                       [0,0,1]]) * scale

    points = apply_T(T_gk, points)
    origin_line = [[0,1], [0,2], [0,3]]
    origin_color = [(1,0,0), (0,1,0), (0,0,1)]

    origin = o3d.geometry.LineSet(
        points= o3d.utility.Vector3dVector(points),
        lines = o3d.utility.Vector2iVector(origin_line),
    )
    origin.colors = o3d.utility.Vector3dVector(origin_color)
    return origin


class MergedDataset(Dataset):
    """
    First dataset type with specific file structure.
    Expected structure:
        data_dir/
        ├── images/
        │   ├── image1.png
        │   └── image2.png
        └── depth_maps/
            ├── image1_depth.png
            └── image2_depth.png
    """
    
    def __init__(
        self,
        data_dir: str,
        bg_dir : str,
        split: str = 'train',
        img_size: Tuple[int, int] = (448, 448),
        augmentation: bool = True,        
    ):
        super().__init__()
        
        # Load file paths
        img_dir = os.path.join(data_dir, "images")
        depth_dir = os.path.join(data_dir, "depths")        
                
        
        img_names = sorted(os.listdir(img_dir))

        split_names = [            
            "IOYS_Fullbody_3D스캔_원본이미지_01",
            "IOYS_Fullbody_3D스캔_원본이미지_02",
            "IOYS_Fullbody_3D스캔_원본이미지_03",
            "IOYS_Fullbody_3D스캔_원본이미지_04",
            "nphm",
            "TH2.1"
        ]        
        
        self.img_paths = []        
        self.calib_paths = []
        self.depth_paths = []
        for split_name in split_names:
            img_root = os.path.join(data_dir, "images", split_name)
            calib_root = os.path.join(data_dir, "calibs", split_name)
            depth_root = os.path.join(data_dir, "depths", split_name)

            subj_names = sorted(os.listdir(img_root))
            
            for subj_name in subj_names:
                img_dir = os.path.join(img_root, subj_name)
                calib_dir = os.path.join(calib_root, subj_name)
                depth_dir = os.path.join(depth_root, subj_name)                
            
                img_names = sorted(os.listdir(img_dir))
                for img_name in img_names:
                    img_path = os.path.join(img_dir, img_name)
                    calib_path = os.path.join(calib_dir, img_name.replace(".jpg", ".json"))
                    depth_path = os.path.join(depth_dir, img_name.replace(".jpg", ".png"))                    

                    if os.path.exists(img_path) and os.path.exists(depth_path) and os.path.exists(calib_path):
                        self.img_paths.append(img_path)
                        self.calib_paths.append(calib_path)
                        self.depth_paths.append(depth_path)

        
        bg_names = sorted(os.listdir(bg_dir))
        self.bg_paths = [os.path.join(bg_dir, bg_name) for bg_name in bg_names]

        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # For validation/test, sample fewer images
        if split != "train":
            self.img_paths = self.img_paths[::200]
            self.calib_paths = self.calib_paths[::200]
            self.depth_paths = self.depth_paths[::200]
        
        assert len(self.img_paths) == len(self.depth_paths)
        assert len(self.depth_paths) == len(self.calib_paths)
        print(f"MergedDataset : Loaded {len(self.img_paths)} samples for {split}")

    def __len__(self):
        return len(self.img_paths)
    
    def load_img(self, img_path, postprocess=True):
        if postprocess:
            img = cv2.imread(img_path)
            img_h, img_w = np.shape(img)[:2]
            mask_path = img_path.replace("images", "masks")
            mask = cv2.imread(mask_path)
            bg_path = np.random.choice(self.bg_paths, 1)[0]
            bg = cv2.imread(bg_path)
            bg = cv2.resize(bg, dsize=(img_w, img_h))

            invalid = mask ==0
            img[invalid] = bg[invalid]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:
            img = Image.open(img_path)
        return img

    def load_depth(self, depth_path):
        ### depth
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 65535 * 10
        return depth

    def load_calib(self, calib_path):
        with open(calib_path, 'r') as json_file:
            calib= json.load(json_file)

        calib["K"] = np.asarray(calib["K"]).astype(np.float32).reshape(3, 3)
        calib["T_gk"] = np.asarray(calib["T_gk"]).astype(np.float32).reshape(4, 4)

        return calib
    
    def normalize_depth(self, depth, calib):
        T_gk = calib["T_gk"]
        K = calib["K"]
        f = K[0, 0]
        cx = K[0, 2]
        scale = f / cx
        T_kg = np.linalg.inv(T_gk)  # merged data's origin is nose tip.

        keys = list(calib.keys())
        if "nose" in keys:
            nose_scan = np.asarray(calib["nose"])  # scale_scan
            nose_scan = np.matmul(T_kg[:3, :3], nose_scan.reshape(3, 1)).transpose(1, 0) + T_kg[:3, -1].reshape(1, 3)

            nose_z = nose_scan.reshape(-1)[-1]
        else:
            nose_z = T_kg[2, -1]

        invalid = depth == 0
        depth = depth / nose_z  # image place 1.0 +-
        depth = depth - 1
        depth *= scale
        depth[invalid] = -1

        return depth
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        calib_path = self.calib_paths[index]
        depth_path = self.depth_paths[index]

        img = self.load_img(img_path, postprocess=True)
        calib = self.load_calib(calib_path)
        depth = self.load_depth(depth_path)
        mask = (depth != 0).astype(np.float32)

        depth_normalized = self.normalize_depth(depth, calib)

        ###
        # pcd = backproject_depth(calib["K"], depth_normalized, np.asarray(img))
        # o3d.visualization.draw_geometries([pcd, make_origin(np.eye(4), scale=1)])
        ###            

        img_tensor = torch.from_numpy(np.asarray(img).astype(np.float32)/255.0).permute(2,0,1) # 3 H W
        mask_tensor = torch.from_numpy(mask).unsqueeze(dim=0) # 1 H W 
        depth_tensor = torch.from_numpy(depth_normalized.astype(np.float32)).unsqueeze(dim=0)

        img_tensor = self.color_jitter(img_tensor)
        img_tensor = self.normalize(img_tensor)

        gts = {
            "depth" : depth_tensor,
            "mask" : mask_tensor,
        }
        return img_tensor, gts 
        



class SynthHumanDataset(Dataset):    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (448, 448),
        augmentation: bool = True,        
    ):
        super().__init__()

        self.img_size = img_size
        self.augmentation = augmentation


        img_dir = os.path.join(data_dir, "images", "faces")
        mask_dir = os.path.join(data_dir, "masks" , "faces")
        depth_dir = os.path.join(data_dir, "depths", "faces")
        calib_dir = os.path.join(data_dir, "calibs", "faces")
        kp_dir = os.path.join(data_dir, "keypoints", "faces")

        kp_names = sorted(os.listdir(kp_dir))        
        if split != "train":
            kp_names = kp_names[::100]
        img_ids = [kp_name.split(".")[0].split("_")[1] for kp_name in kp_names]

        self.img_paths = []
        self.mask_paths = []
        self.depth_paths = []
        self.calib_paths = []
        self.kp_paths = []
        
        self.invalid_paths = []

        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for img_id in img_ids:
            img_path = os.path.join(img_dir, "rgb_"+img_id+".png")
            mask_path = os.path.join(mask_dir, "alpha_"+img_id+".png")
            depth_path = os.path.join(depth_dir, "depth_"+img_id+".exr")
            calib_path = os.path.join(calib_dir, "cam_"+img_id+".txt")
            kp_path = os.path.join(kp_dir, "kps_"+img_id+".npy")            


            if check_existence(img_path, mask_path, depth_path, calib_path, kp_path):
                self.img_paths.append(img_path)
                self.mask_paths.append(mask_path)
                self.depth_paths.append(depth_path)
                self.calib_paths.append(calib_path)
                self.kp_paths.append(kp_path)

        print(f"SynthHumanDataset : Loaded {len(self.img_paths)} samples for {split}")

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        depth_path = self.depth_paths[index]
        calib_path = self.calib_paths[index]
        kp_path = self.kp_paths[index]

        ### raw data preparation
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)/100 # cm
        depth_max = np.max(depth)
        depth[mask==0] = 0.0
        depth[depth==depth_max] = 0.0
        kps = np.load(kp_path)
        kps_int = kps.astype(np.int32)
        nose_depth = depth[kps_int[30, 1], kps_int[30, 0]]
        
        if nose_depth == 0:
            self.invalid_paths.append(kp_path)
        raw_img_h, raw_img_w = np.shape(img)[:2]        
        K = load_txt(calib_path)


        ### preprocessing
        img, mask, depth = pad2square(img, mask, depth)
        img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, dsize=self.img_size, interpolation=cv2.INTER_LANCZOS4)
        depth = cv2.resize(depth, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)

        K[0, 2] =  K[0,2].copy() + int((raw_img_h-raw_img_w)/2)
        K[:2, :] = K[:2, :].copy() * 448 / raw_img_h

        depth_normalized = normalize_depth(K, depth, nose_depth)

        img_tensor = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1) # 3 H W
        mask_tensor = torch.from_numpy(mask.astype(np.float32)/255.0).unsqueeze(dim=0) # 1 H W 
        depth_tensor = torch.from_numpy(depth_normalized.astype(np.float32)).unsqueeze(dim=0)

        img_tensor = self.color_jitter(img_tensor)
        img_tensor = self.normalize(img_tensor)




        # pcd = backproject_depth(K, depth_normalized, img)
        # o3d.visualization.draw_geometries([pcd, make_origin(np.eye(4), scale=1)])

    
        # kps = kps.copy()
        # kps[:,0] += + int((raw_img_h-raw_img_w)/2)
        # kps *= 448/raw_img_h
        # kps_int = kps.astype(np.int32)

        # img[kps_int[:,1], kps_int[:,0]] = (0, 255, 0)
        # img[kps_int[30,1], kps_int[30,0]] = (0, 0, 255)

        # cv2.imshow("vis", img)
        # cv2.waitKey(0)

        gts = {           
            "depth" : depth_tensor,
            "mask" : mask_tensor,
        }
        return img_tensor, gts


class CombinedDepthDataset(Dataset):
    """
    Wrapper to combine multiple datasets with different structures.
    """
    
    def __init__(
        self,
        datasets_config: List[Dict[str, Any]],
        split: str = 'train',
        img_size: Tuple[int, int] = (448, 448),
        augmentation: bool = True
    ):
        """
        Args:
            datasets_config: List of dataset configurations, each containing:
                - 'type': 'dataset1' or 'dataset2'
                - 'path': Path to the dataset
                - 'max_samples': Optional maximum samples from this dataset
                - 'weight': Optional sampling weight for this dataset
        """
        self.datasets = []
        self.dataset_sizes = []
        self.dataset_weights = []
        
        for config in datasets_config:
            dataset_type = config['type'].lower()
            data_dir = config['data_dir']
            bg_dir = config['bg_dir']
            
            weight = config.get('weight', 1.0)
            
            if dataset_type == 'synthhuman':
                dataset = SynthHumanDataset(
                    data_dir=data_dir,
                    split=split,
                    img_size=img_size,
                    augmentation=augmentation,                    
                )
            elif dataset_type == 'merged':
                dataset = MergedDataset(
                    data_dir=data_dir,
                    bg_dir=bg_dir,
                    split=split,
                    img_size=img_size,
                    augmentation=augmentation,                    
                )            
            
            if len(dataset) > 0:
                self.datasets.append(dataset)
                self.dataset_sizes.append(len(dataset))
                self.dataset_weights.append(weight)
        
        if len(self.datasets) == 0:
            raise ValueError("No valid datasets found in the configuration")
        
        # Normalize weights
        total_weight = sum(self.dataset_weights)
        self.dataset_weights = [w / total_weight for w in self.dataset_weights]
        
        # Calculate cumulative sizes for indexing
        self.cumulative_sizes = np.cumsum([0] + self.dataset_sizes)
        self.total_size = self.cumulative_sizes[-1]
        
        print(f"Combined dataset: {len(self.datasets)} datasets, {self.total_size} total samples")
        for i, (dataset, size, weight) in enumerate(zip(self.datasets, self.dataset_sizes, self.dataset_weights)):
            print(f"  Dataset {i}: {size} samples, weight: {weight:.2f}")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        """Get item from the appropriate dataset."""
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        
        # Get the index within that dataset
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        
        # Return the item from the appropriate dataset
        return self.datasets[dataset_idx][local_idx]


class FaceDepthDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Face Depth Estimation task.
    Supports multiple datasets with different structures.
    """
    
    def __init__(
        self,
        datasets_config: Union[List[Dict], Dict[str, List[Dict]]],
        img_size: Tuple[int, int] = (448, 448),
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        augmentation: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Process datasets_config        
        self.datasets_config = datasets_config        
        
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augmentation = augmentation
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None        
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each split."""
        if stage == 'fit' or stage is None:
            # Setup training dataset
            
            self.train_dataset = CombinedDepthDataset(
                datasets_config=self.datasets_config,
                split='train',
                img_size=self.img_size,
                augmentation=self.augmentation
            )
            
            # Setup validation dataset            
            self.val_dataset = CombinedDepthDataset(
                datasets_config=self.datasets_config,
                split='val',
                img_size=self.img_size,
                augmentation=False
            )
        
        if stage == 'test' or stage is None:
            # Setup test dataset
            
            self.test_dataset = CombinedDepthDataset(
                datasets_config=self.datasets_config,
                split='test',
                img_size=self.img_size,
                augmentation=False
            )
    
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_output_channels(self):
        """Return the number of output channels for depth estimation."""
        return 2  # Depth map has 1 channel
    


if __name__ == "__main__":    
    
    dataset = SynthHumanDataset(data_dir="/media/jseob/SSD_HEAD/david/SynthHuman", split="train")

