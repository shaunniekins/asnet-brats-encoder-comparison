# train.py

import tensorflow as tf
from model import AS_Net
from loss import WBEC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
import numpy as np
import os
import datetime
from config import chosen_encoder

start_time = datetime.datetime.now()

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load and preprocess data
train_data = np.load('data/data_train.npy')
train_labels = np.load('data/label_train.npy')
val_data = np.load('data/data_val.npy')
val_labels = np.load('data/label_val.npy')
print("Data loaded")

# Normalize data
train_data = (train_data - train_data.mean()) / train_data.std()
val_data = (val_data - val_data.mean()) / val_data.std()

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels, 4)
val_labels = to_categorical(val_labels, 4)

# Training parameters
batch_size = 32
epochs = 100
steps_per_epoch = len(train_data) // batch_size

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


def generator(images, labels, batch_size):
    while True:
        indices = np.random.permutation(len(images))
        for i in range(0, len(images), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = images[batch_indices]
            batch_labels = labels[batch_indices]
            batch_images_resized = np.array(
                [resize(image, [192, 256]) for image in batch_images])
            batch_images_aug = data_augmentation(batch_images_resized)
            yield batch_images_aug, batch_labels


# Build model
model = AS_Net(encoder=chosen_encoder)
weights_path = f'./checkpoint/{chosen_encoder}_weights.weights.h5'

if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print(f"Loaded weights from {weights_path}.")
else:
    print(f"No weights found at {weights_path}, initializing from scratch.")

# Compile model
optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)
model.compile(optimizer=optimizer, loss=WBEC(), metrics=['accuracy'])

# Callbacks
callbacks = [
    ModelCheckpoint(weights_path, save_best_only=True,
                    monitor='val_loss', mode='min', save_weights_only=True, ),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
    TensorBoard(log_dir='./logs')
]

# Training
history = model.fit(
    generator(train_data, train_labels, batch_size),
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(val_data, val_labels),
    callbacks=callbacks
)

# Save final model
final_model_path = f'training_data/{chosen_encoder}_final_model.h5'
model.save(final_model_path)
print(f"Saved final model at {final_model_path}")
end_time = datetime.datetime.now()

# Calculate duration
duration = end_time - start_time
minutes = divmod(duration.total_seconds(), 60)[0]

# Save training start and end time
training_time_filename = f'training_data/{chosen_encoder}_training_time.txt'
with open(training_time_filename, 'w') as f:
    f.write(f"Start time: {start_time}\n")
    f.write(f"End time: {end_time}\n")
    f.write(f"Duration: {minutes} minutes\n")

print(f"Training start and end times saved to {training_time_filename}")

print("Training completed.")
