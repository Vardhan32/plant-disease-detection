import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set directories
data_dir = "dataset/PlantVillage"
train_dir = 'dataset/PlantVillage'
val_dir = 'dataset/PlantVillage'  # Use entire dataset for training if no split exists


# Hyperparameters
img_size = (128, 128)  # Reduced from (224, 224) for faster training
batch_size = 16  # Reduced batch size
num_classes = len(os.listdir(train_dir))  # Automatically detect number of classes
epochs = 5  # Start with fewer epochs for testing

# Data Augmentation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Split data into 80% train, 20% val

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Use subset to speed up initial training
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load Pretrained Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze pretrained layers

# Custom Classification Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_gen, validation_data=val_gen, epochs=epochs)

# Save Model
model_path = "backend/saved_model/plant_disease_model.h5"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)

print(f"✅ Model trained and saved at: {model_path}")
