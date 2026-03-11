import os
import numpy as np
import rasterio
from skimage.transform import resize


BASE_DATASET = os.path.join(
    "dataset",
    "Dataset of Sentinel-1 SAR and Sentinel-2 NDVI Imagery",
    "Main Folder"
)

VV_DIR = os.path.join(BASE_DATASET, "SAR", "VV")
VH_DIR = os.path.join(BASE_DATASET, "SAR", "VH")
NDVI_DIR = os.path.join(BASE_DATASET, "NDVI")

OUT_IMG = os.path.join("dataset", "dl_training", "images")
OUT_MSK = os.path.join("dataset", "dl_training", "masks")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MSK, exist_ok=True)

# ===================== SANITY CHECKS =====================
print("BASE_DATASET:", BASE_DATASET)
print("BASE_DATASET exists:", os.path.exists(BASE_DATASET))
print("VV_DIR exists:", os.path.exists(VV_DIR))
print("VH_DIR exists:", os.path.exists(VH_DIR))
print("NDVI_DIR exists:", os.path.exists(NDVI_DIR))

if not (os.path.exists(VV_DIR) and os.path.exists(VH_DIR) and os.path.exists(NDVI_DIR)):
    raise FileNotFoundError(
        "❌ Dataset folders not found. "
        "Check BASE_DATASET and folder names carefully."
    )

# ===================== HELPERS =====================
def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

# ===================== MAIN =====================
vv_files = sorted([f for f in os.listdir(VV_DIR) if f.endswith(".tif")])

print(f"Found {len(vv_files)} VV images")

for idx, fname in enumerate(vv_files):
    try:
        vv_path = os.path.join(VV_DIR, fname)
        vh_path = os.path.join(VH_DIR, fname)
        ndvi_path = os.path.join(NDVI_DIR, fname)

        if not (os.path.exists(vh_path) and os.path.exists(ndvi_path)):
            print(f"⚠ Skipping {fname} (missing VH or NDVI)")
            continue

        # 1. Read data
        vv = read_tif(vv_path)
        vh = read_tif(vh_path)
        ndvi = read_tif(ndvi_path)

        # 2. Resize to DL size
        vv = resize(vv, (256, 256), preserve_range=True)
        vh = resize(vh, (256, 256), preserve_range=True)
        ndvi = resize(ndvi, (256, 256), preserve_range=True)

        # 3. Convert SAR to dB
        vv_db = 10 * np.log10(np.maximum(vv, 1e-6))
        vh_db = 10 * np.log10(np.maximum(vh, 1e-6))

        # 4. Compute VV/VH ratio
        ratio = vv_db / (vh_db + 1e-6)

        # 5. Stack channels → (256,256,4)
        image = np.stack([vv_db, vh_db, ratio, ndvi], axis=-1)

        # 6. Weak-supervision flood mask
        mask = (vv_db < -15).astype(np.uint8)

        # 7. Save
        img_out = os.path.join(OUT_IMG, f"img_{idx:03d}.npy")
        msk_out = os.path.join(OUT_MSK, f"mask_{idx:03d}.npy")

        np.save(img_out, image)
        np.save(msk_out, mask)

        print(f"✔ Saved img_{idx:03d}")

    except Exception as e:
        print(f"✖ Error processing {fname}: {e}")

print("✅ DL dataset preparation completed.")
