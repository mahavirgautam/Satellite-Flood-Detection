import numpy as np
import cv2
from skimage.morphology import disk, binary_closing, binary_opening

class FloodChangeDetector:
    @staticmethod
    def log_ratio_change(pre_flood, post_flood, epsilon=1e-10):
        pre_flood_clipped = np.clip(pre_flood, epsilon, None)
        post_flood_clipped = np.clip(post_flood, epsilon, None)
        log_ratio = np.log(post_flood_clipped / pre_flood_clipped)
        return log_ratio
    
    @staticmethod
    def threshold_flood_mask(change_image, method='otsu'):
        if method == 'otsu':
            normalized = ((change_image - np.nanmin(change_image)) / 
                         (np.nanmax(change_image) - np.nanmin(change_image)) * 255).astype(np.uint8)
            threshold_value, binary_mask = cv2.threshold(
                normalized, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            binary_mask = 1 - binary_mask
        else:
            binary_mask = (change_image < -3.0).astype(np.uint8)
        
        return binary_mask
    
    @staticmethod
    def morphological_refinement(flood_mask, kernel_size=5):
        selem = disk(kernel_size)
        flood_mask = binary_closing(flood_mask, selem)
        flood_mask = binary_opening(flood_mask, selem)
        return flood_mask.astype(np.uint8)
    
    @staticmethod
    def compute_flood_metrics(flood_mask, pixel_size_m2=100):
        flood_pixels = np.sum(flood_mask)
        total_pixels = flood_mask.size
        
        flood_area_km2 = (flood_pixels * pixel_size_m2) / 1e6
        flood_percentage = (flood_pixels / total_pixels) * 100
        
        # Determine severity
        if flood_percentage < 5:
            severity = 'Low'
        elif flood_percentage < 15:
            severity = 'Medium'
        elif flood_percentage < 30:
            severity = 'High'
        else:
            severity = 'Critical'
        
        return {
            'flood_area_km2': flood_area_km2,
            'flood_percentage': flood_percentage,
            'flood_pixels': flood_pixels,
            'total_pixels': total_pixels,
            'severity': severity
        }