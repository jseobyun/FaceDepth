import os
import cv2
import argparse
import torch
import open3d as o3d
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from typing import Optional

from src.models import FaceDepthModel
from src.utils.visualization import FaceDepthVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Face Depth Estimation Inference')
    
    parser.add_argument('--input_dir', type=str, default="/home/jseob/Downloads/TEST/dxf_test/images",
                        help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Path to output directory')
    parser.add_argument('--checkpoint', type=str, default="experiments/checkpoints/decoder.ckpt",
                        help='Path to decoder checkpoint (without dinov3)')
    parser.add_argument('--dinov3_checkpoint', type=str, default="checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                        help='Path to dinov3 checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                        help='Input image size (height, width)')    
    parser.add_argument('--save_visualization', action='store_true', default=False,
                        help='Save depth visualization')
    parser.add_argument('--cmap', type=str, default='jet',
                        help='Colormap for depth visualization')
    parser.add_argument('--output_channels', type=int, default=2,
                        help='Number of output channels (depth + mask)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing multiple images')
    
    return parser.parse_args()


class FaceDepthInference:
    """Class for running face depth estimation inference."""
    
    def __init__(
        self,
        checkpoint_path: str,
        dinov3_checkpoint_path: str = None,
        device: str = 'cuda',
        image_size: tuple = (512, 512),
        output_channels: int = 2
    ):
        self.device = torch.device(device)
        self.image_size = image_size
        self.output_channels = output_channels
        
        # Load model
        self.model = self._load_model(checkpoint_path, dinov3_checkpoint_path)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Visualizer
        self.visualizer = FaceDepthVisualizer()
    
    def _load_model(self, checkpoint_path: str, dinov3_checkpoint_path: str = None):
        """Load model from checkpoint with separate dinov3 loading."""
        # Load dinov3 model first
        if dinov3_checkpoint_path:
            print(f"Loading DINOv3 from {dinov3_checkpoint_path}")
            REPO_DIR = "src/models/"
            dinov3 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', 
                                   weights=dinov3_checkpoint_path)
            dinov3 = dinov3.to(self.device)
        else:
            dinov3 = None
            
        # Load decoder checkpoint
        print(f"Loading decoder weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        model = FaceDepthModel(output_channels=self.output_channels)
        
        # Set dinov3 in encoder if provided
        if dinov3 is not None:
            model.encoder.dinov3 = dinov3
            model.encoder.dinov3.eval()
            for p in model.encoder.dinov3.parameters():
                p.requires_grad = False
        
        # Load state dict (decoder and CNN encoder weights)
        if 'state_dict' in checkpoint:
            # Filter out any remaining dinov3 keys if they exist
            state_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                         if 'encoder.dinov3' not in k}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image_path: str):
        """Preprocess input image."""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Apply transforms
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, original_size
    
    def predict(self, image_paths):
        """
        Run inference on images for depth estimation.
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            Predicted depth maps and masks
        """
        # Preprocess images
        image_tensors = []
        original_sizes = []
        for image_path in image_paths:
            image_tensor, original_size = self.preprocess_image(image_path)
            image_tensors.append(image_tensor)
            original_sizes.append(original_size)
        
        image_tensors = torch.cat(image_tensors, dim=0)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensors)
            if isinstance(outputs, dict) and 'final' in outputs:
                pred_output = outputs['final']
            else:
                pred_output = outputs
            
            # Split depth and mask channels
            pred_depth = pred_output[:, :1]  # First channel is depth
            pred_mask = None
            if pred_output.shape[1] > 1:
                pred_mask = torch.sigmoid(pred_output[:, 1:])  # Second channel is mask
        
        return pred_depth, pred_mask, original_sizes
    
    def save_results(
        self,
        image_paths,
        pred_depth: torch.Tensor,
        pred_mask: Optional[torch.Tensor],
        original_sizes: list,
        output_dir: str,
        save_visualization: bool = True,
        cmap: str = 'jet'
    ):
        """Save depth estimation results."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        for img_idx, image_path in enumerate(image_paths):
            base_name = Path(image_path).stem
            
            # Get depth map for this image
            depth_map = pred_depth[img_idx].squeeze()
            
            # Resize to original size if needed
            if original_sizes[img_idx] is not None:
                depth_resized = torch.nn.functional.interpolate(
                    depth_map.unsqueeze(0).unsqueeze(0),
                    size=original_sizes[img_idx][::-1],  # PIL uses (W, H), torch uses (H, W)
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            else:
                depth_resized = depth_map
            
                        
            mask = pred_mask[img_idx].squeeze()
            if original_sizes[img_idx] is not None:
                mask_resized = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=original_sizes[img_idx][::-1],
                    mode='nearest'
                ).squeeze()
            else:
                mask_resized = mask
            
            mask_np = (mask_resized.cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), mask_np)
            
            
                
            # Save side-by-side visualization
            pcd = self.visualizer.visualize_prediction(
                image_path,
                depth_resized,   
                mask_resized,                       
                save_path=os.path.join(output_dir, f"{base_name}_visualization.png")
            )
            
            o3d.visualization.draw_geometries([pcd])
            
            # Save mask if available
            
    
            
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        save_visualization: bool = True,
        cmap: str = 'jet',
        batch_size = 1,
    ):
        """Process all images in a directory."""
        input_path = Path(input_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if len(image_files) == 0:
            print(f"No images found in {input_dir}")
            return
            
        print(f"Found {len(image_files)} images to process")
        print(f"Processing with batch size {batch_size}")       
       
        num_imgs = len(image_files)
        
        # Process each batch with progress tracking
        from tqdm import tqdm
        for img_idx in tqdm(range(0, num_imgs, batch_size), desc="Processing batches"):
            image_paths = image_files[img_idx:img_idx+batch_size]
            try:
                pred_depth, pred_mask, original_sizes = self.predict(image_paths)
                self.save_results(
                    image_paths,
                    pred_depth,
                    pred_mask,
                    original_sizes,
                    output_dir,
                    save_visualization,
                    cmap
                )
            except Exception as e:
                print(f"Error processing batch starting at index {img_idx}: {e}")
                continue
            


def main():
    """Main inference function."""
    args = parse_args()
    
    # Create inference object
    inference = FaceDepthInference(
        checkpoint_path=args.checkpoint,
        dinov3_checkpoint_path=args.dinov3_checkpoint,
        device=args.device,
        image_size=tuple(args.image_size),
        output_channels=args.output_channels
    )
    
    # Check if input is file or directory
    input_path = Path(args.input_dir)
    
    if input_path.is_file():
        # Single image inference
        print(f"Processing single image: {input_path}")
        pred_depth, pred_mask, original_sizes = inference.predict([str(input_path)])
        inference.save_results(
            [str(input_path)],
            pred_depth,
            pred_mask,
            original_sizes,
            args.output_dir,
            args.save_visualization,
            args.cmap
        )
        print(f"Results saved to {args.output_dir}")
    elif input_path.is_dir():
        # Process directory
        inference.process_directory(
            args.input_dir,
            args.output_dir,
            args.save_visualization,
            args.cmap,
            batch_size=args.batch_size,
        )
        print(f"Processing complete. Results saved to {args.output_dir}")
    else:
        raise ValueError(f"Input path {input_path} does not exist")
    


if __name__ == '__main__':
    main()