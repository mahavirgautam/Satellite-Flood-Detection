import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

class SARPreprocessor:
    """SAR-specific preprocessing including calibration, speckle filtering, and dB conversion"""
    
    @staticmethod
    def read_tif(filepath):
        """Read a single TIF file"""
        try:
            with rasterio.open(filepath) as src:
                data = src.read(1).astype(np.float32)
                return data
        except Exception as e:
            raise ValueError(f"Error reading TIF file: {str(e)}")
    
    @staticmethod
    def validate_sar_image(data):
        """Validate SAR image data"""
        if data is None or data.size == 0:
            raise ValueError("Empty image data")
        
        if np.all(data == 0):
            raise ValueError("Image contains only zero values")
        
        if data.shape[0] < 100 or data.shape[1] < 100:
            raise ValueError("Image dimensions too small (minimum 100x100 pixels)")
        
        return True
    
    @staticmethod
    def normalize_to_valid_range(image):
        """Normalize image to valid backscatter range"""
        # Remove invalid values
        valid_mask = np.isfinite(image) & (image != 0)
        
        if not np.any(valid_mask):
            raise ValueError("No valid data in image")
        
        # Get valid data statistics
        valid_data = image[valid_mask]
        
        # If data is already in reasonable range (0-1 or 0-255), scale it
        if np.max(valid_data) <= 1.0:
            # Normalize 0-1 to typical SAR range
            image_normalized = valid_data * 0.3  # Scale to 0-0.3 (typical backscatter)
        elif np.max(valid_data) > 100:
            # Likely DN values (0-255 or 0-65535)
            image_normalized = (valid_data - np.min(valid_data)) / (np.max(valid_data) - np.min(valid_data))
            image_normalized = image_normalized * 0.5  # Scale to 0-0.5
        else:
            # Already in reasonable range
            image_normalized = valid_data / 100.0  # Normalize
        
        # Ensure positive values for dB conversion
        image_normalized = np.clip(image_normalized, 0.0001, 10.0)
        
        # Reconstruct full image
        result = np.zeros_like(image)
        result[valid_mask] = image_normalized
        
        return result
    
    @staticmethod
    def convert_to_db(image, epsilon=1e-10):
        """Convert linear backscatter values to decibel (dB) scale"""
        image_clipped = np.clip(image, epsilon, None)
        db_values = 10 * np.log10(image_clipped)
        
        # SAR backscatter typically ranges from -30 to 10 dB
        # Clip to reasonable range
        db_values = np.clip(db_values, -40, 20)
        
        return db_values
    
    @staticmethod
    def refined_lee_filter(image, window_size=7, cu=0.523):
        """Apply Refined Lee filter for enhanced speckle reduction"""
        mean = uniform_filter(image, window_size)
        sqr_mean = uniform_filter(image**2, window_size)
        variance = sqr_mean - mean**2
        variance = np.maximum(variance, 0)
        
        ci = np.sqrt(variance) / (mean + 1e-10)
        
        w = np.exp(-cu * (ci - 0.5) ** 2)
        w = np.clip(w, 0, 1)
        
        filtered = mean + w * (image - mean)
        
        return filtered
    
    @staticmethod
    def compute_polarization_ratio(vv, vh, epsilon=1e-10):
        """Compute VV/VH cross-polarization ratio"""
        return vv / (vh + epsilon)
    
    @staticmethod
    def process_sar_image(filepath):
        """Complete SAR processing pipeline"""
        # Read image
        data = SARPreprocessor.read_tif(filepath)
        
        # Validate
        SARPreprocessor.validate_sar_image(data)
        
        print(f"Original data range: {np.min(data):.2f} to {np.max(data):.2f}")
        
        # Normalize to valid backscatter range
        normalized = SARPreprocessor.normalize_to_valid_range(data)
        
        print(f"Normalized range: {np.min(normalized):.4f} to {np.max(normalized):.4f}")
        
        # Apply speckle filtering
        filtered = SARPreprocessor.refined_lee_filter(normalized, window_size=7)
        
        # Convert to dB
        data_db = SARPreprocessor.convert_to_db(filtered)
        
        print(f"dB range: {np.min(data_db):.2f} to {np.max(data_db):.2f} dB")
        
        return data_db