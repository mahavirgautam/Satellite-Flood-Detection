import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

class FloodUNet:
    '''U-Net architecture for semantic segmentation of flood areas'''
    
    def __init__(self, input_shape=(256, 256, 4)):
        self.input_shape = input_shape
        self.model = self.build_unet()
    
    def build_unet(self):
        inputs = keras.Input(shape=self.input_shape)
        
        # Encoder
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D(2)(c1)
        
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D(2)(c2)
        
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D(2)(c3)
        
        # Bottleneck
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
        
        # Decoder
        u5 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c4)
        u5 = layers.concatenate([u5, c3])
        c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(u5)
        c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)
        
        u6 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
        u6 = layers.concatenate([u6, c2])
        c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)
        
        u7 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
        u7 = layers.concatenate([u7, c1])
        c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)
        
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)
        
        model = keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', 
                     metrics=['accuracy', tf.keras.metrics.BinaryIoU()])
        
        return model
    
    def prepare_data(self, images, masks, target_size=(256, 256)):
        X_resized = []
        y_resized = []
        
        for img, mask in zip(images, masks):
            img_resized = tf.image.resize(img, target_size).numpy()
            mask_resized = tf.image.resize(mask[..., np.newaxis], target_size).numpy()
            
            X_resized.append(img_resized)
            y_resized.append(mask_resized)
        
        return np.array(X_resized), np.array(y_resized)
    
    def train(self, X, y, epochs=50, batch_size=8, validation_split=0.2):
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(np.uint8)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        
        y_true_flat = y.flatten()
        y_pred_flat = y_pred.flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_true_flat, y_pred_flat),
            'precision': precision_score(y_true_flat, y_pred_flat, zero_division=0),
            'recall': recall_score(y_true_flat, y_pred_flat, zero_division=0),
            'f1_score': f1_score(y_true_flat, y_pred_flat, zero_division=0),
            'iou': jaccard_score(y_true_flat, y_pred_flat, zero_division=0)
        }
        
        return metrics


class RainfallLSTM:
    '''LSTM model for rainfall prediction based on SAR and meteorological features'''
    
    def __init__(self, sequence_length=10, n_features=8):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self.build_lstm()
    
    def build_lstm(self):
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')  # Rainfall in mm
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def extract_temporal_features(self, vv_db, vh_db, ndvi, water_pct):
        '''Extract features for rainfall prediction'''
        features = np.array([
            np.mean(vv_db),
            np.std(vv_db),
            np.mean(vh_db),
            np.std(vh_db),
            np.mean(ndvi) if ndvi is not None else 0,
            np.std(ndvi) if ndvi is not None else 0,
            water_pct,
            np.mean(vv_db / (vh_db + 1e-10))
        ])
        
        return features
    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def predict_rainfall(self, X):
        '''Predict rainfall and confidence'''
        prediction = self.model.predict(X)[0][0]
        
        # Calculate confidence based on model uncertainty
        predictions_multiple = np.array([self.model.predict(X)[0][0] for _ in range(10)])
        confidence = 1 - (np.std(predictions_multiple) / (np.abs(np.mean(predictions_multiple)) + 1e-10))
        confidence = np.clip(confidence * 100, 0, 100)
        
        return max(0, prediction), confidence