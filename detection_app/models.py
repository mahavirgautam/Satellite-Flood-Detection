
from django.db import models
from django.utils import timezone
import uuid

class FloodAnalysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    vv_image = models.FileField(upload_to='uploads/vv/', null=True, blank=True)
    vh_image = models.FileField(upload_to='uploads/vh/', null=True, blank=True)
    rgb_image = models.FileField(upload_to='uploads/rgb/', null=True, blank=True)
    ndvi_image = models.FileField(upload_to='uploads/ndvi/', null=True, blank=True)
    
    upload_date = models.DateTimeField(default=timezone.now)
    processing_status = models.CharField(max_length=50, default='pending')
    
    ml_accuracy = models.FloatField(null=True, blank=True)
    ml_precision = models.FloatField(null=True, blank=True)
    ml_recall = models.FloatField(null=True, blank=True)
    ml_f1_score = models.FloatField(null=True, blank=True)
    ml_iou = models.FloatField(null=True, blank=True)
    best_ml_model = models.CharField(max_length=100, null=True, blank=True)
    
    dl_accuracy = models.FloatField(null=True, blank=True)
    dl_precision = models.FloatField(null=True, blank=True)
    dl_recall = models.FloatField(null=True, blank=True)
    dl_f1_score = models.FloatField(null=True, blank=True)
    dl_iou = models.FloatField(null=True, blank=True)
    
    predicted_rainfall_mm = models.FloatField(null=True, blank=True)
    rainfall_confidence = models.FloatField(null=True, blank=True)
    
    flood_area_km2 = models.FloatField(null=True, blank=True)
    flood_percentage = models.FloatField(null=True, blank=True)
    flood_severity = models.CharField(max_length=50, null=True, blank=True)
    
    flood_mask_image = models.ImageField(upload_to='results/masks/', null=True, blank=True)
    overlay_image = models.ImageField(upload_to='results/overlays/', null=True, blank=True)
    change_detection_image = models.ImageField(upload_to='results/change/', null=True, blank=True)
    
    error_message = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-upload_date']
        verbose_name = 'Flood Analysis'
        verbose_name_plural = 'Flood Analyses'
    
    def __str__(self):
        return f"Analysis {self.id} - {self.upload_date.strftime('%Y-%m-%d %H:%M')}"
    
    def get_severity_color(self):
        severity_map = {
            'Low': 'success',
            'Medium': 'warning',
            'High': 'danger',
            'Critical': 'danger'
        }
        return severity_map.get(self.flood_severity, 'secondary')