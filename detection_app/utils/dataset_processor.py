import os
import numpy as np
import pandas as pd
import rasterio
from django.conf import settings
from .sar_processor import SARPreprocessor
from .flood_detector import FloodChangeDetector

class DatasetMatcher:
    """Match uploaded images with similar samples from training dataset"""
    
    def __init__(self):
        self.dataset_path = settings.DATASET_BASE_PATH
        self.vv_path = os.path.join(self.dataset_path, 'Main Folder', 'SAR', 'VV')
        self.vh_path = os.path.join(self.dataset_path, 'Main Folder', 'SAR', 'VH')
        self.metadata_path = os.path.join(self.dataset_path, 'Main Folder', 'DATA.xlsx')
        
        # Load metadata once
        self.metadata = pd.read_excel(self.metadata_path)
        self.processor = SARPreprocessor()
    
    def find_similar_samples(self, vv_uploaded, vh_uploaded, n_samples=3):
        """Find similar samples from dataset based on backscatter statistics"""
        
        # Calculate statistics of uploaded image
        vv_mean = np.nanmean(vv_uploaded)
        vv_std = np.nanstd(vv_uploaded)
        vh_mean = np.nanmean(vh_uploaded)
        vh_std = np.nanstd(vh_uploaded)
        
        # Get all dataset samples
        dataset_files = sorted([f for f in os.listdir(self.vv_path) if f.endswith('.tif')])
        
        similarities = []
        
        for i, filename in enumerate(dataset_files[:18]):  # Use all 18 samples
            try:
                # Load dataset sample
                vv_dataset = self.processor.process_sar_image(os.path.join(self.vv_path, filename))
                vh_dataset = self.processor.process_sar_image(os.path.join(self.vh_path, filename))
                
                # Calculate similarity score (lower is better)
                similarity = (
                    abs(np.nanmean(vv_dataset) - vv_mean) +
                    abs(np.nanstd(vv_dataset) - vv_std) +
                    abs(np.nanmean(vh_dataset) - vh_mean) +
                    abs(np.nanstd(vh_dataset) - vh_std)
                )
                
                similarities.append({
                    'index': i,
                    'filename': filename,
                    'similarity': similarity,
                    'water_pct': self.metadata.iloc[i]['Water percentage Mean'],
                    'vegetation_pct': self.metadata.iloc[i]['Vegetation percentage Mean'],
                    'cloudiness': self.metadata.iloc[i]['Cloudiness Mean']
                })
                
            except Exception as e:
                continue
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'])
        
        return similarities[:n_samples]
    
    def estimate_flood_from_dataset(self, vv_uploaded, vh_uploaded):
        """Estimate flood metrics based on similar dataset samples"""
        
        similar_samples = self.find_similar_samples(vv_uploaded, vh_uploaded, n_samples=5)
        
        if not similar_samples:
            raise ValueError("No similar samples found in dataset")
        
        # Calculate weighted average of water percentages
        total_weight = sum([1.0 / (s['similarity'] + 0.1) for s in similar_samples])
        
        weighted_water_pct = sum([
            s['water_pct'] * (1.0 / (s['similarity'] + 0.1)) / total_weight
            for s in similar_samples
        ])
        
        # Calculate statistics from uploaded image
        vv_threshold_pixels = np.sum(vv_uploaded < -12)  # Water threshold
        total_pixels = vv_uploaded.size
        detected_water_pct = (vv_threshold_pixels / total_pixels) * 100
        
        # Blend dataset estimate with direct detection
        final_water_pct = (weighted_water_pct * 0.6 + detected_water_pct * 0.4)
        
        return {
            'water_percentage': final_water_pct,
            'similar_samples': similar_samples,
            'dataset_estimate': weighted_water_pct,
            'direct_detection': detected_water_pct
        }