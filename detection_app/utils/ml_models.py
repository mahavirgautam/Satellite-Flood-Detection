import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

class FastFloodClassifier:
    """Fast classifier using sampled pixels"""
    
    def __init__(self, sample_size=5000):
        self.sample_size = sample_size
        self.model = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
    
    def extract_features(self, vv_db, vh_db, ratio_db, ndvi=None):
        """Extract pixel-level features"""
        if ndvi is not None:
            features = np.stack([vv_db, vh_db, ratio_db, ndvi], axis=-1)
        else:
            features = np.stack([vv_db, vh_db, ratio_db], axis=-1)
        
        return features.reshape(-1, features.shape[-1])
    
    def train_on_sample(self, X, y):
        """Train on sampled pixels for speed"""
        
        # Sample pixels
        if len(X) > self.sample_size:
            indices = np.random.choice(len(X), self.sample_size, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
        
        # Handle NaN
        X_sample = np.nan_to_num(X_sample, nan=0.0)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_sample)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_sample, test_size=0.2, random_state=42, stratify=y_sample
        )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'iou': jaccard_score(y_test, y_pred, zero_division=0)
        }
        
        return metrics
    
    def predict_full_image(self, X):
        """Predict on full image"""
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)