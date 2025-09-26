import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from typing import Dict, Any, Optional
from .encoder import Encoder
from .decoder import Decoder

class FaceDepthModel(pl.LightningModule):
    """
    PyTorch Lightning module for Face Depth Estimation task.
    """
    
    def __init__(
        self,
        output_channels: int = 2,  # Depth map has 1 channel
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Network architecture will be defined here
        self.encoder = Encoder()
        self.decoder = Decoder(output_channels=output_channels)
        
        # Loss functions for depth estimation
        self.l1_loss = nn.L1Loss(reduction='none')
        self.l2_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss()
                
    def forward(self, x):
        """Forward pass through the network."""
        # Placeholder implementation
        features = self.encoder(x)
        output = self.decoder(features)
        return output
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     targets: torch.Tensor, 
                     aux_weight: float = 0.4) -> Dict[str, torch.Tensor]:               
        
        gt_mask = targets["mask"]
        # Main depth loss (combination of L1 and L2)
        l1_loss = self.l1_loss(outputs["final"][:, :1], targets["depth"])[gt_mask >0.99]
        l2_loss = self.l2_loss(outputs["final"][:, :1], targets["depth"])[gt_mask >0.99]
        l1_loss = l1_loss.mean()
        l2_loss = l2_loss.mean()
        main_loss = l1_loss + 0.5 * l2_loss  # Weighted combination

        ce_loss = self.bce_loss(outputs["final"][:, 1:], targets["mask"])
        
        # Auxiliary losses for deep supervision
        total_loss = main_loss + 0.01 * ce_loss
        aux_loss = 0
        
        if 'auxs' in outputs and outputs['auxs']:
            for aux_out in outputs['auxs'].values():
                aux_l1 = self.l1_loss(aux_out[:, :1], targets["depth"])[gt_mask >0.99]
                aux_l2 = self.l2_loss(aux_out[:, :1], targets["depth"])[gt_mask >0.99]
                aux_l1 = aux_l1.mean()
                aux_l2 = aux_l2.mean()
                aux_ce = self.bce_loss(aux_out[:, 1:], targets["mask"])
                aux_loss += (aux_l1 + 0.5 * aux_l2 + 0.01 * ce_loss)
            
            aux_loss = aux_loss / len(outputs['auxs'])
            total_loss = total_loss + aux_weight * aux_loss
        
        return total_loss
            
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, targets = batch
        
        # Forward pass
        outputs = self(images)
        
        # Calculate loss
        loss = self.compute_loss(outputs, targets)                
        self.log("train/loss", loss)                               
                
        return {"loss" : loss}
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""        
        images, targets = batch
        
        # Forward pass
        outputs = self(images)
        
        # Calculate loss
        loss = self.compute_loss(outputs, targets)                
        self.log("val/loss", loss)                                
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.scheduler_config is None:
            return optimizer
        
        # Example scheduler configuration
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.scheduler_config.get('T_max', 100),
            eta_min=self.hparams.scheduler_config.get('eta_min', 1e-6)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }