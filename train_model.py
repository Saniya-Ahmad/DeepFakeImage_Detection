import os
import sys
import glob
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# Paths and Constants
# ===============================
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5

EXCEL_FILES = {
    "train_real": os.path.join(DATASET_DIR, "train", "real.xlsx"),
    "train_fake": os.path.join(DATASET_DIR, "train", "fake.xlsx"),
    "val_real":   os.path.join(DATASET_DIR, "valid", "real.xlsx"),
    "val_fake":   os.path.join(DATASET_DIR, "valid", "fake.xlsx"),
    "test_real":  os.path.join(DATASET_DIR, "test", "real.xlsx"),
    "test_fake":  os.path.join(DATASET_DIR, "test", "fake.xlsx"),
}

# ===============================
# Helper Functions
# ===============================

def read_pair_excel(real_xlsx, fake_xlsx):
    """Read real/fake Excel files and return combined DataFrame with 'path' and 'label'."""
    try:
        real_df = pd.read_excel(real_xlsx)
        fake_df = pd.read_excel(fake_xlsx)
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel(s): {e}")
    
    # Use 'path' or fallback to 'original_path'
    for col in ["path", "original_path"]:
        if col in real_df.columns and col in fake_df.columns:
            path_col = col
            break
    else:
        raise KeyError("Excel files must contain 'path' or 'original_path' column.")
    
    real_df = real_df.copy()
    fake_df = fake_df.copy()
    real_df["label"] = 0
    fake_df["label"] = 1
    
    real_df["path"] = real_df[path_col].astype(str).str.strip()
    fake_df["path"] = fake_df[path_col].astype(str).str.strip()
    
    df = pd.concat([real_df, fake_df], ignore_index=True)
    return df

def candidate_files_for_stem(folder, stem):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    found = []
    for ext in exts:
        f = os.path.join(folder, stem + ext)
        if os.path.exists(f):
            found.append(f)
    found += glob.glob(os.path.join(folder, stem + ".*"))
    return found

def resolve_path(val):
    """Resolve a path from Excel to an actual file."""
    if val is None: return None
    s = str(val).strip()
    if not s: return None
    if os.path.isabs(s) and os.path.exists(s):
        return s
    cand = os.path.normpath(os.path.join(BASE_DIR, s))
    if os.path.exists(cand): return cand
    cand2 = os.path.normpath(os.path.join(DATASET_DIR, s))
    if os.path.exists(cand2): return cand2
    folder = os.path.dirname(cand2)
    stem = os.path.splitext(os.path.basename(s))[0]
    if os.path.isdir(folder):
        matches = candidate_files_for_stem(folder, stem)
        if matches: return os.path.normpath(matches[0])
    folder2 = os.path.dirname(cand)
    if os.path.isdir(folder2):
        matches = candidate_files_for_stem(folder2, stem)
        if matches: return os.path.normpath(matches[0])
    for ext in ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff"]:
        found = glob.glob(os.path.join(DATASET_DIR, "**", stem + ext), recursive=True)
        if found: return os.path.normpath(found[0])
    return None

def load_images_from_df(df, max_missing_print=40):
    X, y = [], []
    missing = 0
    missing_examples = []
    for _, row in df.iterrows():
        p = resolve_path(row.get("path"))
        if p is None:
            missing += 1
            if len(missing_examples) < max_missing_print:
                missing_examples.append(str(row.get("path")))
            continue
        try:
            img = load_img(p, target_size=IMG_SIZE)
            arr = img_to_array(img) / 255.0
            X.append(arr)
            y.append(int(row["label"]))
        except Exception as e:
            missing += 1
            if len(missing_examples) < max_missing_print:
                missing_examples.append(f"{p} (error: {e})")
    for ex in missing_examples:
        print("❌ Missing / unreadable:", ex)
    if missing > len(missing_examples):
        print(f"...and {missing - len(missing_examples)} more missing/unreadable paths")
    print(f"✅ Loaded {len(X)} images, skipped {missing} rows")
    return np.array(X), np.array(y, dtype=np.int32)

def build_cnn_feature_extractor(input_shape=(*IMG_SIZE, 3)):
    inp = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation="relu")(inp)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation="relu")(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    model = Model(inputs=inp, outputs=x, name="cnn_feature_extractor")
    return model

# ===============================
# Main
# ===============================
def main():
    # Check Excel files exist
    for k, p in EXCEL_FILES.items():
        if not os.path.exists(p):
            print(f"❗ Missing Excel file: {p}")
            sys.exit(1)
    
    # Read datasets
    train_df = read_pair_excel(EXCEL_FILES["train_real"], EXCEL_FILES["train_fake"])
    val_df   = read_pair_excel(EXCEL_FILES["val_real"], EXCEL_FILES["val_fake"]) if os.path.exists(EXCEL_FILES["val_real"]) else pd.DataFrame(columns=["path","label"])
    test_df  = read_pair_excel(EXCEL_FILES["test_real"], EXCEL_FILES["test_fake"]) if os.path.exists(EXCEL_FILES["test_real"]) else pd.DataFrame(columns=["path","label"])
    
    # Load images
    X_train, y_train = load_images_from_df(train_df)
    X_val, y_val     = load_images_from_df(val_df) if not val_df.empty else (np.empty((0,*IMG_SIZE,3)), np.empty((0,)))
    X_test, y_test   = load_images_from_df(test_df) if not test_df.empty else (np.empty((0,*IMG_SIZE,3)), np.empty((0,)))

    print("\nDataset summary -> Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))
    if len(X_train) == 0:
        print("❌ No training images loaded. Check 'original_path' in your Excel files.")
        sys.exit(1)

    # Build CNN
    feature_cnn = build_cnn_feature_extractor()
    classifier_output = Dense(1, activation="sigmoid")(feature_cnn.output)
    classifier_model = Model(inputs=feature_cnn.input, outputs=classifier_output)

    classifier_model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    classifier_model.summary()

    # Train
    classifier_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val)>0 else None,
        epochs=EPOCHS,
        batch_size=min(BATCH_SIZE, max(1, len(X_train)))
    )

    # Save CNN feature extractor
    feat_path = os.path.join(MODEL_DIR, "cnn_feature_extractor.h5")
    feature_cnn.save(feat_path)
    print("💾 Saved CNN feature extractor:", feat_path)

    # Extract features for SVM
    X_train_feat = feature_cnn.predict(X_train)
    X_val_feat   = feature_cnn.predict(X_val) if len(X_val)>0 else np.empty((0, X_train_feat.shape[1]))
    X_test_feat  = feature_cnn.predict(X_test) if len(X_test)>0 else np.empty((0, X_train_feat.shape[1]))

    # Train SVM
    svm = SVC(kernel="rbf", probability=True)
    svm.fit(X_train_feat, y_train)
    svm_path = os.path.join(MODEL_DIR, "svm_model.pkl")
    joblib.dump(svm, svm_path)
    print("💾 Saved SVM model:", svm_path)

    # Evaluate
    if X_val_feat.shape[0] > 0:
        y_val_pred = svm.predict(X_val_feat)
        print("Validation accuracy:", accuracy_score(y_val, y_val_pred))
    if X_test_feat.shape[0] > 0:
        y_test_pred = svm.predict(X_test_feat)
        print("Test accuracy:", accuracy_score(y_test, y_test_pred))
        print("\nClassification Report (test):")
        print(classification_report(y_test, y_test_pred))

if __name__ == "__main__":
    main()
