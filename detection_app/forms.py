
from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Row, Column, Div
from .models import FloodAnalysis

class FloodImageUploadForm(forms.ModelForm):
    class Meta:
        model = FloodAnalysis
        fields = ['vv_image', 'vh_image', 'rgb_image', 'ndvi_image']
        widgets = {
            'vv_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.tif,.tiff',
                'required': True
            }),
            'vh_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.tif,.tiff',
                'required': True
            }),
            'rgb_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.tif,.tiff',
                'required': False
            }),
            'ndvi_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.tif,.tiff',
                'required': False
            }),
        }
        labels = {
            'vv_image': 'SAR VV Polarization Image (Required)',
            'vh_image': 'SAR VH Polarization Image (Required)',
            'rgb_image': 'RGB Optical Image (Optional)',
            'ndvi_image': 'NDVI Image (Optional)',
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_enctype = 'multipart/form-data'
        self.helper.layout = Layout(
            Div(
                Row(
                    Column('vv_image', css_class='form-group col-md-6 mb-3'),
                    Column('vh_image', css_class='form-group col-md-6 mb-3'),
                ),
                Row(
                    Column('rgb_image', css_class='form-group col-md-6 mb-3'),
                    Column('ndvi_image', css_class='form-group col-md-6 mb-3'),
                ),
                css_class='card-body'
            ),
            Div(
                Submit('submit', 'Analyze Flood Images', css_class='btn btn-primary btn-lg w-100 mt-3'),
                css_class='card-body pt-0'
            )
        )
    
    def clean(self):
        cleaned_data = super().clean()
        vv_image = cleaned_data.get('vv_image')
        vh_image = cleaned_data.get('vh_image')
        
        if not vv_image or not vh_image:
            raise forms.ValidationError("VV and VH polarization images are required!")
        
        # Validate file extensions
        for field_name in ['vv_image', 'vh_image', 'rgb_image', 'ndvi_image']:
            file = cleaned_data.get(field_name)
            if file:
                if not file.name.lower().endswith(('.tif', '.tiff')):
                    raise forms.ValidationError(f"{field_name}: Only .tif or .tiff files are allowed!")
                
                # Check file size (max 100MB)
                if file.size > 100 * 1024 * 1024:
                    raise forms.ValidationError(f"{field_name}: File size exceeds 100MB limit!")
        
        return cleaned_data