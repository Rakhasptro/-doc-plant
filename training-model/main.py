# ====================================================================
# SISTEM PAKAR DIAGNOSA PENYAKIT TANAMAN - VERSI PERBAIKAN KERAS 3
# ====================================================================

# ====================================================================
# STEP 1: Setup Environment & Install Dependencies
# ====================================================================
print("📦 Installing dependencies...")

# Install dengan versi terbaru
!pip install -q tensorflowjs==4.17.0
!pip install -q kaggle
!pip install -q pillow

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import zipfile
from google.colab import files

print("✅ Dependencies installed!")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# ====================================================================
# STEP 2: Upload Kaggle API Key
# ====================================================================
print("\n" + "="*60)
print("📤 UPLOAD KAGGLE.JSON")
print("="*60)

# UNCOMMENT UNTUK UPLOAD KAGGLE.JSON
uploaded = files.upload()

# Setup Kaggle credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

print("✅ Kaggle API Key berhasil di-setup!")

# ====================================================================
# STEP 3: Download Dataset dari Kaggle
# ====================================================================
print("\n" + "="*60)
print("📥 DOWNLOADING DATASET")
print("="*60)

# Download dataset (gunakan dataset alternatif jika perlu)
try:
    !kaggle datasets download -d emmarex/plantdisease
    dataset_zip = 'plantdisease.zip'
except Exception as e:
    print(f"⚠️ Error download: {e}")
    print("Coba dataset alternatif...")
    !kaggle datasets download -d vipoooool/new-plant-diseases-dataset
    dataset_zip = 'new-plant-diseases-dataset.zip'

# Extract dataset
print("\n📦 Extracting dataset...")
!unzip -q {dataset_zip} -d plant_dataset

# Bersihkan file tersembunyi
print("\n🧹 Membersihkan file non-image...")
!find plant_dataset -type f -name '.*' -delete
!find plant_dataset -type f -name '*.txt' -delete

print("✅ Dataset berhasil di-download dan extract!")

# Check dataset structure
print("\n📂 Struktur Dataset:")
!ls plant_dataset/

# Tentukan path yang benar
if os.path.exists('plant_dataset/PlantVillage'):
    dataset_path = 'plant_dataset/PlantVillage'
elif os.path.exists('plant_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'):
    dataset_path = 'plant_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
elif os.path.exists('plant_dataset/train'):
    dataset_path = 'plant_dataset/train'
else:
    # Cari folder dengan banyak subdirektori (class folders)
    for root, dirs, files in os.walk('plant_dataset'):
        if len(dirs) > 10:  # Jika ada banyak subfolder (kelas)
            dataset_path = root
            break

print(f"✅ Dataset path: {dataset_path}")

# ====================================================================
# STEP 4: Data Preprocessing
# ====================================================================
print("\n" + "="*60)
print("🔄 DATA PREPROCESSING")
print("="*60)

# Setup parameters
img_size = (224, 224)
batch_size = 16  # Dikurangi untuk menghindari OOM

# Data augmentation untuk training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Load training data
print("📊 Loading training data...")
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
print("📊 Loading validation data...")
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get class information
num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

print(f"\n✅ Data berhasil di-load!")
print(f"📈 Total kelas penyakit: {num_classes}")
print(f"📈 Total gambar training: {train_generator.samples}")
print(f"📈 Total gambar validasi: {validation_generator.samples}")

# Simpan class names
with open('class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)

print("\n🏷️ Sample kelas penyakit:")
for i, name in enumerate(class_names[:5]):
    print(f"  {i+1}. {name}")
if num_classes > 5:
    print(f"  ... dan {num_classes - 5} kelas lainnya")

# Visualize sample images
print("\n📸 Visualisasi sample data:")
sample_datagen = ImageDataGenerator(rescale=1./255)
sample_generator = sample_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=9,
    class_mode='categorical',
    shuffle=True
)

plt.figure(figsize=(15, 8))
sample_batch = next(sample_generator)
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(sample_batch[0][i])
    class_idx = np.argmax(sample_batch[1][i])
    plt.title(class_names[class_idx], fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Hapus sample generator
del sample_generator, sample_datagen

# ====================================================================
# STEP 5: Build CNN Model
# ====================================================================
print("\n" + "="*60)
print("🏗️ BUILDING CNN MODEL")
print("="*60)

# Clear previous models
keras.backend.clear_session()

# Build model dengan MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# Build full model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model berhasil dibuat!")
print("\n📊 Ringkasan Model:")
model.summary()

# ====================================================================
# STEP 6: Train Model
# ====================================================================
print("\n" + "="*60)
print("🎓 TRAINING MODEL")
print("="*60)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU tersedia: {'✅ Ya' if gpus else '⚠️ Tidak (menggunakan CPU)'}")

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.keras',  # Gunakan .keras untuk Keras 3
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train model
print("\n🚀 Mulai training...\n")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training selesai!")

# ====================================================================
# STEP 7: Evaluate Model
# ====================================================================
print("\n" + "="*60)
print("📊 EVALUASI MODEL")
print("="*60)

# Get final accuracy
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\n📈 Training Accuracy: {final_train_acc*100:.2f}%")
print(f"📈 Validation Accuracy: {final_val_acc*100:.2f}%")
print(f"📉 Training Loss: {final_train_loss:.4f}")
print(f"📉 Validation Loss: {final_val_loss:.4f}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Training', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ====================================================================
# STEP 8: Convert to TensorFlow.js (KERAS 3 COMPATIBLE)
# ====================================================================
print("\n" + "="*60)
print("🔄 CONVERT TO TENSORFLOW.JS (KERAS 3)")
print("="*60)

# Verifikasi model terbaik ada
if not os.path.exists('best_model.keras'):
    print("⚠️ best_model.keras tidak ditemukan, menggunakan model terakhir")
    model.save('best_model.keras')

# Bersihkan folder sebelumnya
print("🧹 Membersihkan folder konversi...")
!rm -rf tfjs_model saved_model_export

# Load model terbaik
print("📂 Loading best model...")
model = keras.models.load_model('best_model.keras')
print("✅ Model loaded")

# SOLUSI KERAS 3: Gunakan model.export() untuk SavedModel
saved_model_path = 'saved_model_export'

try:
    print("💾 Exporting model dalam SavedModel format (Keras 3)...")

    # Keras 3 menggunakan export() bukan save()
    model.export(saved_model_path)

    print("✅ Model berhasil di-export")

    # Verifikasi struktur
    print("\n📂 Struktur SavedModel:")
    !ls -lh {saved_model_path}

    # Convert ke TensorFlow.js
    print("\n🔄 Converting ke TensorFlow.js...")
    !tensorflowjs_converter \
        --input_format=tf_saved_model \
        --output_format=tfjs_graph_model \
        --signature_name=serving_default \
        --saved_model_tags=serve \
        {saved_model_path} \
        tfjs_model

    print("✅ Konversi berhasil!")

except Exception as e:
    print(f"⚠️ Error dengan model.export(): {e}")
    print("\n🔧 Mencoba metode alternatif dengan tf.saved_model.save()...")

    try:
        # Metode alternatif: Buat wrapper function
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)])
        def serving_fn(input_tensor):
            return {'output': model(input_tensor, training=False)}

        # Save dengan signature
        tf.saved_model.save(
            model,
            saved_model_path,
            signatures={'serving_default': serving_fn}
        )

        print("✅ Model berhasil disimpan dengan metode alternatif")

        # Convert ke TensorFlow.js
        print("\n🔄 Converting ke TensorFlow.js...")
        !tensorflowjs_converter \
            --input_format=tf_saved_model \
            --output_format=tfjs_graph_model \
            --signature_name=serving_default \
            --saved_model_tags=serve \
            {saved_model_path} \
            tfjs_model

        print("✅ Konversi berhasil!")

    except Exception as e2:
        print(f"❌ Metode alternatif juga gagal: {e2}")
        print("\n🔧 Mencoba metode terakhir: Direct Python API...")

        try:
            import tensorflowjs as tfjs

            # Simpan model dalam format h5 dulu
            model.save('temp_model.h5')

            # Load ulang sebagai legacy model
            legacy_model = keras.models.load_model('temp_model.h5')

            # Convert dengan Python API
            tfjs.converters.save_keras_model(legacy_model, 'tfjs_model')

            print("✅ Konversi dengan Python API berhasil!")

        except Exception as e3:
            print(f"❌ Semua metode gagal: {e3}")
            raise

# Verifikasi hasil
print("\n🔍 Verifikasi hasil konversi:")
if os.path.exists('tfjs_model/model.json'):
    print("✅ model.json ditemukan!")
    print("\n📂 File yang dihasilkan:")
    !ls -lh tfjs_model/

    # Zip model
    print("\n📦 Membuat ZIP file...")
    !zip -r plant_disease_model.zip tfjs_model class_names.json
    print("✅ ZIP berhasil dibuat!")

    # Cek ukuran file
    print("\n📊 Ukuran file:")
    !du -sh plant_disease_model.zip

    # ====================================================================
    # STEP 9: Download Model
    # ====================================================================
    print("\n" + "="*60)
    print("📥 DOWNLOAD MODEL")
    print("="*60)

    files.download('plant_disease_model.zip')
    files.download('class_names.json')

    print("\n🎉 SELESAI!")
    print("="*60)

else:
    print("❌ KONVERSI GAGAL! File model.json tidak ditemukan")
    print("\n📋 Periksa folder yang ada:")
    !ls -la tfjs_model/ 2>/dev/null || echo "Folder tfjs_model tidak ada"

# ====================================================================
# BONUS: Test Prediction
# ====================================================================
print("\n" + "="*60)
print("🧪 TEST PREDICTION")
print("="*60)

try:
    # Reset validation generator
    validation_generator.reset()

    # Get sample image
    val_batch = next(validation_generator)
    sample_img = val_batch[0][0:1]
    sample_label = val_batch[1][0]

    # Predict
    prediction = model.predict(sample_img, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx] * 100

    actual_class_idx = np.argmax(sample_label)
    actual_class = class_names[actual_class_idx]

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.imshow(sample_img[0])
    plt.title(f'Actual: {actual_class}\nPredicted: {predicted_class} ({confidence:.2f}%)',
              fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.show()

    print(f"\n🎯 Prediksi: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print(f"✅ Actual: {actual_class}")
    print(f"{'✅ BENAR!' if predicted_class == actual_class else '❌ SALAH'}")

except Exception as e:
    print(f"⚠️ Test prediction error: {e}")

print("\n" + "="*60)
print("✅ SEMUA PROSES SELESAI!")
print("="*60)
print("\n📝 Catatan:")
print("- Model telah dikonversi ke TensorFlow.js")
print("- File dapat digunakan di web browser")
print("- Load model dengan: tf.loadGraphModel('model.json')")