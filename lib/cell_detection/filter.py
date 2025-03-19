from torchvision import transforms
import torch
import cv2
import numpy as np
from typing import List, Tuple

def filter_candidate_cells(original_img: np.ndarray, 
                         mask_img: np.ndarray, 
                         model: torch.nn.Module, 
                         device: torch.device,
                         transform=None, 
                         L: int=20,
                         batch_size: int=128):
    """
    Filter candidate cells using batch processing when GPU is available
    
    Args:
        original_img: Original image
        mask_img: Binary mask image
        model: PyTorch model for classification
        device: torch.device for computation
        transform: Optional transform pipeline
        L: Size of the square crop
        batch_size: Batch size for GPU processing
    """
    model.to(device)
    if transform is None:
        transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((L, L)), 
            transforms.ToTensor()
        ])
    
    # Get connected components
    num_labels, labels = cv2.connectedComponents(mask_img)
    processed_mask = np.zeros_like(mask_img)
    half_L = int(L / 2)
    
    # If using CPU, keep original processing
    if device.type == 'cpu':
        for i in range(1, num_labels):
            ys, xs = np.where(labels == i)
            centroid_x, centroid_y = int(np.mean(xs)), int(np.mean(ys))
            
            x1, y1 = max(centroid_x - half_L, 0), max(centroid_y - half_L, 0)
            x2, y2 = min(centroid_x + half_L, original_img.shape[1]), min(centroid_y + half_L, original_img.shape[0])

            cropped_img = original_img[y1:y2, x1:x2]
            cropped_img = cv2.resize(cropped_img, (L, L))
            cropped_img = transform(cropped_img).unsqueeze(0).to(device)
            
            output = model(cropped_img)
            _, predicted = torch.max(output.data, 1)
            
            if predicted.item() == 1:
                processed_mask[ys, xs] = 255
                
        return processed_mask
    
    # For GPU, use batch processing
    # First collect all crops and their coordinates
    crops = []
    coordinates = []
    
    for i in range(1, num_labels):
        ys, xs = np.where(labels == i)
        centroid_x, centroid_y = int(np.mean(xs)), int(np.mean(ys))
        
        x1, y1 = max(centroid_x - half_L, 0), max(centroid_y - half_L, 0)
        x2, y2 = min(centroid_x + half_L, original_img.shape[1]), min(centroid_y + half_L, original_img.shape[0])

        cropped_img = original_img[y1:y2, x1:x2]
        cropped_img = cv2.resize(cropped_img, (L, L))
        crops.append(transform(cropped_img))
        coordinates.append((ys, xs))
    
    # Process in batches
    with torch.no_grad():  # Add no_grad for inference
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            batch_tensor = torch.stack(batch_crops).to(device)
            
            outputs = model(batch_tensor)
            _, predictions = torch.max(outputs.data, 1)
            
            # Update mask for this batch
            for pred, (ys, xs) in zip(predictions, coordinates[i:i + batch_size]):
                if pred.item() == 1:
                    processed_mask[ys, xs] = 255
    
    return processed_mask
