import torch
import numpy as np
import open3d as o3d
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import cv2


class FaceDepthVisualizer:
    """Utility class for visualizing face depth estimation results."""
    
    @staticmethod
    def depth_to_colormap(depth: np.ndarray, cmap: str = 'jet', vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
        """
        Convert depth map to color image using a colormap.
        
        Args:
            depth: Depth map of shape (H, W)
            cmap: Matplotlib colormap name
            vmin: Minimum value for normalization
            vmax: Maximum value for normalization
            
        Returns:
            Color image of shape (H, W, 3)
        """
        if vmin is None:
            vmin = depth.min()
        if vmax is None:
            vmax = depth.max()
        
        # Normalize depth to [0, 1]
        depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
        depth_normalized = np.clip(depth_normalized, 0, 1)
        
        # Apply colormap
        cm = plt.get_cmap(cmap)
        depth_colored = cm(depth_normalized)
        
        # Convert to uint8 RGB
        depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)
        
        return depth_colored
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert torch tensor to numpy array.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Numpy array
        """
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Handle different tensor shapes
        if len(tensor.shape) == 4:  # Batch dimension
            tensor = tensor[0]
        
        if len(tensor.shape) == 3:  # Channel dimension
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            elif tensor.shape[0] == 3:
                # Denormalize if needed (ImageNet normalization)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor = tensor * std + mean
                tensor = tensor.clamp(0, 1)
                tensor = tensor.permute(1, 2, 0)
        
        return tensor.numpy()
    
    @staticmethod
    def visualize_prediction(
        image_path: str,
        pred_depth: torch.Tensor,    
        pred_mask: torch.Tensor,            
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize depth prediction results.
        
        Args:
            image_path: Path to input image
            pred_depth: Predicted depth tensor
            gt_depth: Ground truth depth tensor (optional)
            cmap: Colormap for depth visualization
            save_path: Path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        # Convert to numpy        
        pred_depth = FaceDepthVisualizer.tensor_to_numpy(pred_depth)
        pred_mask = FaceDepthVisualizer.tensor_to_numpy(pred_mask)
        # Get depth colormaps        
        img_h, img_w = pred_depth.shape[:2]
        
        # Load and resize input image
        image_np = cv2.imread(image_path)        
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, dsize=(img_w, img_h))        

        u, v = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h), indexing="xy")
        cx = img_w/2
        cy = img_h/2
        f = img_w/2

        x = (u-cx) / f
        y = (v-cy) / f
        z = pred_depth        
        points = np.concatenate([x[:,:,None], y[:,:,None], z[:,:,None]], axis=-1)
        valid = pred_mask >0.3
        colors = image_np[valid].reshape(-1, 3).astype(np.float32)/255.0
        points = points[valid].reshape(-1, 3)

        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
        pcd.colors = o3d.utility.Vector3dVector(colors)


        return pcd
    
    @staticmethod
    def save_depth_map(depth: np.ndarray, save_path: str, normalize: bool = True):
        """
        Save depth map as image file.
        
        Args:
            depth: Depth map array
            save_path: Path to save the depth map
            normalize: Whether to normalize depth values to [0, 255]
        """
        if normalize:
            depth_min = depth.min()
            depth_max = depth.max()
            depth_normalized = ((depth - depth_min) / (depth_max - depth_min + 1e-8) * 255).astype(np.uint8)
        else:
            depth_normalized = (depth * 255).astype(np.uint8)
        
        Image.fromarray(depth_normalized).save(save_path)
    
    @staticmethod
    def create_colorbar(cmap: str = 'jet', save_path: Optional[str] = None) -> np.ndarray:
        """
        Create a colorbar legend for depth visualization.
        
        Args:
            cmap: Colormap name
            save_path: Path to save colorbar
            
        Returns:
            Colorbar as numpy array
        """
        fig, ax = plt.subplots(figsize=(6, 1))
        
        # Create gradient
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack([gradient] * 20)
        
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_xticklabels(['Near', '', 'Mid', '', 'Far'])
        ax.set_yticks([])
        ax.set_title('Depth Scale', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        # Convert to numpy array
        fig.canvas.draw()
        colorbar_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        colorbar_array = colorbar_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        
        return colorbar_array