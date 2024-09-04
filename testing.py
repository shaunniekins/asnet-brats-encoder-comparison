from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


def evaluate_model(model, tr_gen, valid_gen, ts_gen):
    train_score = model.evaluate(tr_gen, verbose=1)
    valid_score = model.evaluate(valid_gen, verbose=1)
    test_score = model.evaluate(ts_gen, verbose=1)

    print(f"Train Loss: {train_score[0]:.4f}")
    print(f"Train Accuracy: {train_score[1]*100:.2f}%")
    print('-' * 20)
    print(f"Validation Loss: {valid_score[0]:.4f}")
    print(f"Validation Accuracy: {valid_score[1]*100:.2f}%")
    print('-' * 20)
    print(f"Test Loss: {test_score[0]:.4f}")
    print(f"Test Accuracy: {test_score[1]*100:.2f}%")

    preds = model.predict(ts_gen)
    y_pred = np.argmax(preds, axis=1)

    cm = confusion_matrix(ts_gen.classes, y_pred)
    labels = list(tr_gen.class_indices.keys())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('Truth Label')
    plt.show()

    clr = classification_report(ts_gen.classes, y_pred)
    print(clr)

    # Save the classification report to a file
    with open('classification_report.txt', 'w') as file:
        file.write(clr)


def predict(model, img_path, class_dict):
    label = list(class_dict.keys())
    plt.figure(figsize=(12, 12))
    img = Image.open(img_path)
    resized_img = img.resize((299, 299))
    img = np.asarray(resized_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    predictions = model.predict(img)
    probs = list(predictions[0])
    labels = label
    plt.subplot(2, 1, 1)
    plt.imshow(resized_img)
    plt.subplot(2, 1, 2)
    bars = plt.barh(labels, probs)
    plt.xlabel('Probability', fontsize=15)
    ax = plt.gca()
    ax.bar_label(bars, fmt='%.2f')
    plt.show()
