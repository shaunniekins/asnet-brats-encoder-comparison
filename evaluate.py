# evaluate.py

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from model import AS_Net
from config import chosen_encoder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.set_printoptions(threshold=np.inf)

# Load Data
val_data = np.load('data/data_val.npy').astype(np.float32)
val_labels = np.load('data/label_val.npy').astype(np.int32)
label_classes = np.load('data/label_classes.npy', allow_pickle=True)

val_data = tf.image.adjust_gamma(val_data / 255., gamma=1.6)

print('Brain Tumor MRI Dataset loaded')

model = AS_Net(encoder=chosen_encoder)
model.load_weights = f'./checkpoint/{chosen_encoder}_weights.weights.h5'
predictions = model.predict(val_data, batch_size=1, verbose=1)

y_pred = np.argmax(predictions, axis=-1)
y_true = val_labels

output_folder = 'output/'
os.makedirs(output_folder, exist_ok=True)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_classes))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(label_classes))
plt.xticks(tick_marks, label_classes, rotation=45)
plt.yticks(tick_marks, label_classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(output_folder + f'{chosen_encoder}_confusion_matrix.png')

# Save the results
with open(output_folder + f'{chosen_encoder}_performances.txt', 'w') as file_perf:
    file_perf.write("Classification Report:\n")
    file_perf.write(classification_report(y_true, y_pred, target_names=label_classes))
    file_perf.write("\nConfusion Matrix:\n")
    file_perf.write(str(cm))

# Save sample results
fig, ax = plt.subplots(16, 2, figsize=[15, 48])

for idx in range(min(16, len(val_data))):
    ax[idx, 0].imshow(val_data[idx])
    ax[idx, 0].set_title(f"True: {label_classes[y_true[idx]]}")
    ax[idx, 1].imshow(val_data[idx])
    ax[idx, 1].set_title(f"Predicted: {label_classes[y_pred[idx]]}")

plt.tight_layout()
plt.savefig(output_folder + f'{chosen_encoder}_sample_results.png')

print("Evaluation completed. Results saved in the 'output' folder.")