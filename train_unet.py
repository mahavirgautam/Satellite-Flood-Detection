import os
import numpy as np
from detection_app.utils.deep_learning import FloodUNet

# ================= PATHS =================
IMAGE_DIR = "dataset/dl_training/images"
MASK_DIR  = "dataset/dl_training/masks"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "unet_flood.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= LOAD DATA =================
images = []
masks = []

image_files = sorted(os.listdir(IMAGE_DIR))
print(f"Found {len(image_files)} training samples")

for fname in image_files:
    img = np.load(os.path.join(IMAGE_DIR, fname))
    mask = np.load(os.path.join(MASK_DIR, fname.replace("img", "mask")))

    images.append(img)
    masks.append(mask[..., np.newaxis])

X = np.array(images, dtype=np.float32)
y = np.array(masks, dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ================= MODEL =================
unet = FloodUNet(input_shape=(256, 256, 4))

# ================= TRAIN =================
unet.train(
    X, y,
    epochs=10,          # reasonable
    batch_size=2,
    validation_split=0.2
)

# ================= SAVE =================
unet.model.save(MODEL_PATH)

print("✅ Training completed.")
print(f"✅ Model saved at: {MODEL_PATH}")
