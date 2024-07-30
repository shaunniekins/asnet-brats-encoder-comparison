# contents

- [Dataset Preparation](#prepare_dataset)
- [Config](#config)
- [Loss](#loss)
- [Train](#train)
- [Model](#model)
- [Evaluate](#evaluate)

<!--
can you explain this code but by function and important lines? i want to grasp what each lines do.  also explain what does the imported function do (eg. train_test_split, LabelEncoder, etc.):
 -->

<!-- 1 -->

## prepare_dataset

[↑](#contents)

### Imported Modules and Their Functions

1. **`numpy as np`**: Provides support for large, multi-dimensional arrays and matrices. It's used here for array manipulations and saving data.

2. **`os`**: Provides a way to interact with the operating system, such as file and directory management. It is used for path operations.

3. **`pandas as pd`**: A data manipulation and analysis library that uses dataframes, which are 2D labeled data structures. It’s used here to handle file paths and labels in a structured way.

4. **`Image` from `PIL`**: Part of the Python Imaging Library, it’s used to open and manipulate images.

5. **`train_test_split` from `sklearn.model_selection`**: Splits data arrays into two subsets: for training data and validation data. It helps in creating training and validation sets from a single dataset.

6. **`LabelEncoder` from `sklearn.preprocessing`**: Converts categorical labels into numeric form. It is useful for preparing data for machine learning algorithms.

7. **`ThreadPoolExecutor` and `as_completed` from `concurrent.futures`**: Facilitates concurrent (multi-threaded) execution of tasks. It is used to process images in parallel to speed up the data loading process.

### Functions and Key Lines

1. **`generate_data_paths(data_dir)`**:

   - **Purpose**: Creates a DataFrame containing file paths and corresponding labels for all images in the dataset directory.
   - **Key Lines**:
     ```python
     folds = os.listdir(data_dir)
     for fold in folds:
         foldpath = os.path.join(data_dir, fold)
         if os.path.isdir(foldpath):
             filelist = os.listdir(foldpath)
             for file in filelist:
                 fpath = os.path.join(foldpath, file)
                 filepaths.append(fpath)
                 labels.append(fold)
     return pd.DataFrame({'filepaths': filepaths, 'labels': labels})
     ```
     - `os.listdir(data_dir)`: Lists all subdirectories (folds) in the dataset directory.
     - `os.path.join(data_dir, fold)`: Constructs the path to each fold.
     - `pd.DataFrame(...)`: Creates a DataFrame with file paths and labels.

2. **`encode_labels(df)`**:

   - **Purpose**: Encodes categorical labels into numeric form and adds this encoding to the DataFrame.
   - **Key Lines**:
     ```python
     le = LabelEncoder()
     df['label_encoded'] = le.fit_transform(df['labels'])
     return df, le
     ```
     - `LabelEncoder()`: Initializes the label encoder.
     - `le.fit_transform(df['labels'])`: Converts categorical labels to numeric values.

3. **`process_image(filepath)`**:

   - **Purpose**: Loads an image, resizes it, and converts it to a NumPy array.
   - **Key Lines**:
     ```python
     img = Image.open(filepath).convert('RGB')
     img = np.array(img.resize((width, height))).astype(np.float32)
     return img
     ```
     - `Image.open(filepath).convert('RGB')`: Opens and converts the image to RGB format.
     - `img.resize((width, height))`: Resizes the image to the specified dimensions.
     - `np.array(...).astype(np.float32)`: Converts the image to a NumPy array of type float32.

4. **`load_data_in_batches(df)`**:

   - **Purpose**: Loads images in parallel, processes them, and returns the processed images and their corresponding labels.
   - **Key Lines**:
     ```python
     with ThreadPoolExecutor() as executor:
         futures = [executor.submit(process_image, filepath)
                    for filepath in df['filepaths']]
         for future in as_completed(futures):
             image = future.result()
             if image is not None:
                 images.append(image)
     return np.array(images), np.array(df['label_encoded'])
     ```
     - `ThreadPoolExecutor()`: Creates a thread pool for parallel processing.
     - `executor.submit(process_image, filepath)`: Submits the image processing tasks to the thread pool.
     - `as_completed(futures)`: Iterates over completed tasks as they finish.

5. **Main Script Execution (`if __name__ == "__main__":`)**:
   - **Purpose**: Executes the main workflow of generating paths, encoding labels, splitting the dataset, loading images, and saving processed data.
   - **Key Lines**:
     ```python
     train_df = generate_data_paths(train_dir)
     train_df, label_encoder = encode_labels(train_df)
     train_df, val_df = train_test_split(
         train_df, test_size=0.2, stratify=train_df['labels'], random_state=42)
     Data_train, Label_train = load_data_in_batches(train_df)
     Data_val, Label_val = load_data_in_batches(val_df)
     np.save(os.path.join(save_dir, 'data_train'), Data_train)
     np.save(os.path.join(save_dir, 'label_train'), Label_train)
     np.save(os.path.join(save_dir, 'data_val'), Data_val)
     np.save(os.path.join(save_dir, 'label_val'), Label_val)
     np.save(os.path.join(save_dir, 'label_classes'), label_encoder.classes_)
     ```
     - `generate_data_paths(train_dir)`: Generates file paths and labels.
     - `encode_labels(train_df)`: Encodes labels.
     - `train_test_split(...)`: Splits the dataset into training and validation sets.
     - `load_data_in_batches(...)`: Loads and processes images in batches.
     - `np.save(...)`: Saves the processed data and labels to disk.

This script is designed to prepare a dataset of MRI images for training and validation in a machine learning model. It involves generating file paths, encoding labels, processing images, splitting data, and saving the results.

<!-- 2 -->

## config

[↑](#contents)

### Purpose

The script is used to handle command line arguments to specify the type of encoder to be used in a machine learning model. It allows the user to choose between different encoder architectures (`vgg16`, `mobilenetv3`, `efficientnetv2`) when running the script.

### Key Components

1. **Importing the `argparse` Module**:

   ```python
   import argparse
   ```

   - **Purpose**: The `argparse` module provides a way to handle command line arguments. It helps parse arguments provided by the user when running the script.

2. **Creating the Argument Parser**:

   ```python
   parser = argparse.ArgumentParser(description='Choose an encoder for the model.')
   ```

   - **Purpose**: Initializes an `ArgumentParser` object which will be used to parse command line arguments. The `description` parameter provides a brief explanation of what the script does.

3. **Adding an Argument**:

   ```python
   parser.add_argument('--encoder', type=str, choices=['vgg16', 'mobilenetv3', 'efficientnetv2'], default='vgg16', help='Encoder type')
   ```

   - **Purpose**: Defines an argument that the script accepts.
   - **Parameters**:
     - `--encoder`: The name of the command line argument (in this case, `--encoder`).
     - `type=str`: Specifies that the argument should be a string.
     - `choices=['vgg16', 'mobilenetv3', 'efficientnetv2']`: Limits the acceptable values to these specific options. This ensures that only these choices can be provided.
     - `default='vgg16'`: Specifies the default value to use if the user does not provide this argument.
     - `help='Encoder type'`: Provides a short description of what the argument is for, which is shown when the user requests help (e.g., by running `python script.py --help`).

4. **Parsing the Arguments**:

   ```python
   args = parser.parse_args()
   ```

   - **Purpose**: Parses the command line arguments provided by the user and stores them in the `args` object. This object contains the values of the arguments.

5. **Configuring the Encoder**:
   ```python
   chosen_encoder = args.encoder
   ```
   - **Purpose**: Retrieves the value of the `--encoder` argument from the `args` object and assigns it to the `chosen_encoder` variable. This variable now holds the encoder type chosen by the user or the default value if no argument was provided.

### Example Usage

To use this script, you would run it from the command line and provide the `--encoder` argument like this:

```bash
python script.py --encoder mobilenetv3
```

If you don't provide the `--encoder` argument, the script will use the default value (`vgg16`).

### Summary

This script allows users to specify which encoder architecture they want to use for a machine learning model through command line arguments. It uses the `argparse` module to handle argument parsing, and provides a choice of three encoders (`vgg16`, `mobilenetv3`, `efficientnetv2`) with a default option. This approach makes the script flexible and user-friendly for different configurations.

<!-- 3 -->

## loss

[↑](#contents)

### Import Statements

1. **`import tensorflow as tf`**:

   - **Purpose**: Imports the TensorFlow library, which provides the core functionality for building and training machine learning models.

2. **`from keras import backend as K`**:

   - **Purpose**: Imports the backend functions from Keras, which provide various utility functions for TensorFlow/Keras operations.

3. **`from keras.losses import Loss`**:
   - **Purpose**: Imports the `Loss` class function from Keras. The `Loss` class is used to create custom loss functions.

### Custom Loss Function Class

1. **`class WBEC(Loss):`**:

   - **Purpose**: Defines a custom loss class called `WBEC` that inherits from Keras' `Loss` class. This class will implement a custom loss function.

2. **`def __init__(self, weight=2.5):`**:

   - **Purpose**: Initializes the custom loss function with a `weight` parameter. This weight is a hyperparameter that will be used to scale the loss contribution of positive examples.
   - **Key Line**: `self.weight = weight` stores the weight as an instance variable.

3. **`def call(self, y_true, y_pred):`**:
   - **Purpose**: Defines the computation of the custom loss function. This method is called during model training to compute the loss based on true labels (`y_true`) and predicted labels (`y_pred`).
4. **`y_pred = tf.convert_to_tensor(y_pred)`**:

   - **Purpose**: Converts `y_pred` to a TensorFlow tensor to ensure compatibility with TensorFlow operations.

5. **`y_true = tf.cast(y_true, y_pred.dtype)`**:

   - **Purpose**: Casts `y_true` to the same data type as `y_pred` to avoid type mismatch issues.

6. **`epslion_ = tf.constant(K.epsilon(), y_pred.dtype.base_dtype)`**:

   - **Purpose**: Defines a small constant (`epsilon`) to prevent log of zero during loss computation. `K.epsilon()` provides a small constant value (e.g., `1e-7`), and it is cast to the data type of `y_pred`.

7. **`y_pred = tf.clip_by_value(y_pred, epslion_, 1.0 - epslion_)`**:

   - **Purpose**: Clips the values of `y_pred` to be within the range `[epsilon, 1 - epsilon]` to avoid taking the logarithm of zero or very close to zero, which could lead to numerical instability.

8. **`wbce = self.weight * y_true * tf.math.log(y_pred + K.epsilon())`**:

   - **Purpose**: Computes the weighted binary cross-entropy loss for positive samples (`y_true` is 1). It multiplies the log loss by the weight for positive samples.

9. **`wbce += (1 - y_true) * tf.math.log(1 - y_pred + K.epsilon())`**:

   - **Purpose**: Adds the binary cross-entropy loss for negative samples (`y_true` is 0). This computes the log loss for the `(1 - y_true)` part and adds it to `wbce`.

10. **`return -wbce`**:
    - **Purpose**: Returns the negative of the computed loss, as TensorFlow/Keras loss functions expect a value to be minimized. The negative sign is used because TensorFlow minimizes the loss function during training.

### Summary

The `WBEC` class defines a custom loss function for binary classification that includes a weighting mechanism for positive examples. Here’s a brief summary of how it works:

1. **Initialization**: Sets up the weight parameter for the positive class.
2. **Loss Calculation**: Converts inputs to tensors, clips predictions to avoid numerical issues, and computes the weighted binary cross-entropy loss. The loss is returned as a negative value because TensorFlow minimizes the loss function.

This custom loss function can be used in model training to adjust the importance of positive examples in the loss calculation, which can be useful in scenarios with class imbalance.

<!-- 4 -->

## train

[↑](#contents)

### Import Statements

1. **`import tensorflow as tf`**:

   - **Purpose**: Imports TensorFlow, which is used for building and training the machine learning model.

2. **`from model import AS_Net`**:

   - **Purpose**: Imports the custom model architecture `AS_Net` from the `model` module.

3. **`from loss import WBEC`**:

   - **Purpose**: Imports the custom loss function `WBEC` from the `loss` module.

4. **`from tensorflow.keras.optimizers import Adam`**:

   - **Purpose**: Imports the Adam optimizer from TensorFlow’s Keras module for optimizing the model during training.

5. **`from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard`**:

   - **Purpose**: Imports various Keras callbacks used to monitor and adjust the training process:
     - `ModelCheckpoint`: Saves model weights during training.
     - `EarlyStopping`: Stops training early if the model performance does not improve.
     - `ReduceLROnPlateau`: Reduces the learning rate when a plateau in the validation loss is detected.
     - `TensorBoard`: Logs data for visualization in TensorBoard.

6. **`from tensorflow.keras.utils import to_categorical`**:

   - **Purpose**: Imports a utility function to convert class vectors to one-hot encoded format.

7. **`from tensorflow.image import resize`**:

   - **Purpose**: Imports the image resizing function from TensorFlow.

8. **`import numpy as np`**:

   - **Purpose**: Imports NumPy for numerical operations and handling arrays.

9. **`import os`**:

   - **Purpose**: Provides functions to interact with the operating system, such as file path management.

10. **`import datetime`**:

    - **Purpose**: Provides classes for manipulating dates and times.

11. **`from config import chosen_encoder`**:
    - **Purpose**: Imports the `chosen_encoder` variable from a configuration module. This specifies which encoder architecture to use for the model.

### Script Workflow

1. **GPU Setup**:

   ```python
   gpus = tf.config.experimental.list_physical_devices('GPU')
   if gpus:
       try:
           for gpu in gpus:
               tf.config.experimental.set_memory_growth(gpu, True)
       except RuntimeError as e:
           print(e)
   ```

   - **Purpose**: Configures TensorFlow to use GPU(s) with memory growth enabled. This helps manage GPU memory allocation to avoid running out of memory.

2. **Load and Preprocess Data**:

   ```python
   train_data = np.load('data/data_train.npy')
   train_labels = np.load('data/label_train.npy')
   val_data = np.load('data/data_val.npy')
   val_labels = np.load('data/label_val.npy')
   print("Data loaded")
   ```

   - **Purpose**: Loads training and validation data and labels from disk.

   **Normalization**:

   ```python
   train_data = (train_data - np.mean(train_data, axis=(1, 2, 3), keepdims=True)) / np.std(train_data, axis=(1, 2, 3), keepdims=True)
   val_data = (val_data - np.mean(val_data, axis=(1, 2, 3), keepdims=True)) / np.std(val_data, axis=(1, 2, 3), keepdims=True)
   ```

   - **Purpose**: Normalizes each image by subtracting its mean and dividing by its standard deviation to standardize the input data.

   **Convert Labels to One-Hot Encoding**:

   ```python
   train_labels = to_categorical(train_labels, 4)
   val_labels = to_categorical(val_labels, 4)
   ```

   - **Purpose**: Converts class labels to one-hot encoded format suitable for classification tasks with 4 classes.

3. **Training Parameters**:

   ```python
   batch_size = 32
   epochs = 100
   steps_per_epoch = len(train_data) // batch_size
   ```

   - **Purpose**: Defines the batch size, number of epochs, and steps per epoch for training.

4. **Data Augmentation**:

   ```python
   data_augmentation = tf.keras.Sequential([
       tf.keras.layers.RandomFlip("horizontal"),
       tf.keras.layers.RandomRotation(0.1),
       tf.keras.layers.RandomZoom(0.1),
   ])
   ```

   - **Purpose**: Defines a sequence of data augmentation techniques to apply to training images to improve model generalization.

5. **Data Generator**:

   ```python
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
   ```

   - **Purpose**: Defines a generator function that yields batches of augmented images and labels. This function handles shuffling, resizing, and augmentation of the data.

6. **Build Model**:

   ```python
   model = AS_Net(encoder=chosen_encoder)
   weights_path = f'./checkpoint/{chosen_encoder}_weights.weights.h5'

   if os.path.exists(weights_path):
       model.load_weights(weights_path)
       print(f"Loaded weights from {weights_path}.")
   else:
       print(f"No weights found at {weights_path}, initializing from scratch.")
   ```

   - **Purpose**: Initializes the model with the specified encoder architecture. It tries to load pre-trained weights if they exist, or initializes the model from scratch if they do not.

7. **Compile Model**:

   ```python
   optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)
   model.compile(optimizer=optimizer, loss=WBEC(), metrics=['accuracy'])
   ```

   - **Purpose**: Compiles the model with the Adam optimizer, custom `WBEC` loss function, and accuracy as the performance metric.

8. **Callbacks**:

   ```python
   callbacks = [
       ModelCheckpoint(weights_path, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True),
       EarlyStopping(patience=10, restore_best_weights=True),
       ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
       TensorBoard(log_dir='./logs')
   ]
   ```

   - **Purpose**: Defines a list of callbacks for model training:
     - `ModelCheckpoint`: Saves the best model weights based on validation loss.
     - `EarlyStopping`: Stops training if no improvement is seen for 10 epochs and restores the best weights.
     - `ReduceLROnPlateau`: Reduces the learning rate when the validation loss plateaus.
     - `TensorBoard`: Logs training metrics for visualization in TensorBoard.

9. **Training**:

   ```python
   history = model.fit(
       generator(train_data, train_labels, batch_size),
       epochs=epochs,
       steps_per_epoch=steps_per_epoch,
       validation_data=(val_data, val_labels),
       callbacks=callbacks
   )
   ```

   - **Purpose**: Trains the model using the defined data generator, number of epochs, and callbacks.

10. **Save Final Model**:

    ```python
    final_model_path = f'training_data/{chosen_encoder}_final_model.h5'
    model.save(final_model_path)
    print(f"Saved final model at {final_model_path}")
    ```

    - **Purpose**: Saves the trained model to disk.

11. **Save Training Time**:

    ```python
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    minutes = divmod(duration.total_seconds(), 60)[0]

    training_time_filename = f'training_data/{chosen_encoder}_training_time.txt'
    with open(training_time_filename, 'w') as f:
        f.write(f"Start time: {start_time}\n")
        f.write(f"End time: {end_time}\n")
        f.write(f"Duration: {minutes} minutes\n")

    print(f"Training start and end times saved to {training_time_filename}")
    ```

    - **Purpose**: Calculates and saves the start time, end time, and duration of the training process.

### Summary

The `train.py` script performs the following tasks:

1. **Sets Up GPU**: Configures TensorFlow to use available GPUs with memory growth enabled.
2. **Loads and Preprocesses Data**: Loads, normalizes, and converts data to one-hot encoding.
3. **Defines Training Parameters and Data Augmentation**: Sets batch size, epochs, and data augmentation techniques.
4. **Creates Data Generator**: Defines a generator for feeding data to the model during training.
5. **Builds and Compiles Model**: Initializes the model, loads weights if available, and compiles it with the Adam optimizer and custom loss function.
6. **Sets Up Callbacks**: Configures callbacks for model checkpointing, early stopping, learning rate reduction, and TensorBoard logging.
7. **Trains Model**: Trains the model using the defined parameters and callbacks.
8. **Saves Model and Training Information**: Saves the final model and training

<!-- 5 -->

## model

[↑](#contents)

### Import Statements

1. **`from tensorflow.keras import Model, Input`**:

   - **Purpose**: Imports the base `Model` class and `Input` layer from TensorFlow Keras, which are used to define and build the model.

2. **`from tensorflow.keras.applications import VGG16, EfficientNetV2L, MobileNetV3Large`**:

   - **Purpose**: Imports pre-trained models (VGG16, EfficientNetV2L, MobileNetV3Large) from Keras applications to use as encoders in the model.

3. **`from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate, Multiply, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape`**:

   - **Purpose**: Imports various Keras layers for constructing the neural network (e.g., convolutional layers, pooling layers, concatenation, etc.).

4. **`from tensorflow.keras import backend as K`**:
   - **Purpose**: Imports the backend module from Keras, which provides utility functions for TensorFlow/Keras operations.

### `AS_Net` Function

The `AS_Net` function defines a custom neural network architecture:

1. **Input Layer**:

   ```python
   inputs = Input(input_size)
   ```

   - **Purpose**: Defines the input layer of the model with a specified shape.

2. **Encoder Selection**:

   ```python
   if encoder == 'vgg16':
       ENCODER = VGG16(weights='imagenet', include_top=False, input_shape=input_size)
       layer_indices = [2, 5, 9, 13, 17]
   elif encoder == 'mobilenetv3':
       ENCODER = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=input_size)
       layer_indices = [0, 3, 6, 10, 15]
   elif encoder == 'efficientnetv2':
       ENCODER = EfficientNetV2L(weights='imagenet', include_top=False, input_shape=input_size)
       layer_indices = [2, 6, 12, 19, 32]
   else:
       raise ValueError("Unsupported encoder type. Choose from 'vgg16', 'mobilenetv3', or 'efficientnetv2'.")
   ```

   - **Purpose**: Chooses a pre-trained encoder model (VGG16, MobileNetV3Large, or EfficientNetV2L) based on the input `encoder` parameter. The `layer_indices` list defines which layers' outputs will be used.

3. **Feature Extraction**:

   ```python
   output_layers = [ENCODER.get_layer(index=i).output for i in layer_indices]
   outputs = [Model(inputs=ENCODER.inputs, outputs=layer)(inputs) for layer in output_layers]
   ```

   - **Purpose**: Extracts feature maps from the specified layers of the chosen encoder.

4. **Feature Map Adjustment and Merging**:

   ```python
   def adjust_feature_map(x, target_shape):
       _, h, w, _ = target_shape
       current_h, current_w = x.shape[1:3]
       if current_h > h or current_w > w:
           return MaxPooling2D(pool_size=(current_h // h, current_w // w))(x)
       elif current_h < h or current_w < w:
           return UpSampling2D(size=(h // current_h, w // current_w))(x)
       return x

   merged = outputs[-1]
   for i in range(len(outputs) - 2, -1, -1):
       adjusted = adjust_feature_map(outputs[i], merged.shape)
       merged = concatenate([merged, adjusted], axis=-1)
   ```

   - **Purpose**: Adjusts the feature map sizes using pooling or upsampling to match the size of the largest feature map and then concatenates them.

5. **SAM and CAM Modules**:

   ```python
   SAM1 = SAM(filters=merged.shape[-1])(merged)
   CAM1 = CAM(filters=merged.shape[-1])(merged)

   combined = concatenate([SAM1, CAM1], axis=-1)
   ```

   - **Purpose**: Applies the custom `SAM` and `CAM` modules to the merged feature map and concatenates their outputs.

6. **Final Layers**:

   ```python
   output = Conv2D(64, 3, activation='relu', padding='same')(combined)
   output = GlobalAveragePooling2D()(output)
   output = Dense(4, activation='softmax', kernel_initializer='he_normal')(output)
   ```

   - **Purpose**: Applies a convolutional layer, followed by global average pooling and a dense layer with softmax activation for classification.

7. **Model Creation**:
   ```python
   model = Model(inputs=inputs, outputs=output)
   return model
   ```
   - **Purpose**: Creates and returns the Keras model instance using the defined architecture.

### `SAM` Class

The `SAM` (Selective Attention Module) class defines a custom attention mechanism:

1. **Initialization**:

   ```python
   def __init__(self, filters):
       super(SAM, self).__init__()
       self.filters = filters
       self.conv1 = Conv2D(self.filters // 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
       self.conv2 = Conv2D(self.filters // 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
       self.conv3 = Conv2D(self.filters // 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
       self.conv4 = Conv2D(self.filters // 4, 1, activation='relu', kernel_initializer='he_normal')
       self.W1 = Conv2D(self.filters // 4, 1, activation='sigmoid', kernel_initializer='he_normal')
       self.W2 = Conv2D(self.filters // 4, 1, activation='sigmoid', kernel_initializer='he_normal')
   ```

   - **Purpose**: Initializes convolutional layers and attention weights for the SAM module.

2. **Forward Pass**:

   ```python
   def call(self, inputs):
       out1 = self.conv3(self.conv2(self.conv1(inputs)))
       out2 = self.conv4(inputs)

       pool1 = GlobalAveragePooling2D()(out2)
       pool1 = Reshape((1, 1, self.filters // 4))(pool1)
       merge1 = self.W1(pool1)

       pool2 = GlobalMaxPooling2D()(out2)
       pool2 = Reshape((1, 1, self.filters // 4))(pool2)
       merge2 = self.W2(pool2)

       out3 = merge1 + merge2
       y = Multiply()([out1, out3]) + out2
       return y
   ```

   - **Purpose**: Computes attention maps and applies them to feature maps to emphasize important regions.

### `CAM` Class

The `CAM` (Channel Attention Module) class defines another custom attention mechanism:

1. **Initialization**:

   ```python
   def __init__(self, filters, reduction_ratio=16):
       super(CAM, self).__init__()
       self.filters = filters
       self.conv1 = Conv2D(self.filters // 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
       self.conv2 = Conv2D(self.filters // 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
       self.conv3 = Conv2D(self.filters // 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
       self.conv4 = Conv2D(self.filters // 4, 1, activation='relu', kernel_initializer='he_normal')
       self.gpool = GlobalAveragePooling2D()
       self.fc1 = Dense(self.filters // (4 * reduction_ratio), activation='relu', use_bias=False)
       self.fc2 = Dense(self.filters // 4, activation='sigmoid', use_bias=False)
   ```

   - **Purpose**: Initializes convolutional and dense layers for channel-wise attention.

2. **Forward Pass**:
   ```python
   def call(self, inputs):
       out1 = self.conv3(self.conv2(self.conv1(inputs)))
       out2 = self.conv4(inputs)
       out3 = self.fc2(self.fc1(self.gpool(out2)))
       out3 = Reshape((1, 1, self.filters // 4))(out3)
       y = Multiply()([out1, out3]) + out2
       return y
   ```
   - **Purpose**: Applies channel attention to highlight important channels in the feature maps.

### `Synergy` Class

The `Synergy` class defines a module for combining features:

1. **Initialization**:

   ```python
   def __init__(self, alpha=0.5, beta=0.5):
       super(Synergy, self).__init__()
       self.alpha = alpha
       self.beta = beta
       self.conv = Conv2D(1, 3, padding='same', kernel_initializer='he_normal')
       self.bn = BatchNormalization()


   ```

   - **Purpose**: Initializes a convolutional layer and batch normalization for combining features with adjustable weights.

2. **Forward Pass**:
   ```python
   def call(self, inputs):
       x, y = inputs
       inputs = self.alpha * x + self.beta * y
       y = self.bn(self.conv(inputs))
       return y
   ```
   - **Purpose**: Combines two input feature maps using weighted summation and applies normalization.

### Main Block

```python
if __name__ == '__main__':
    print(K.epsilon())
```

- **Purpose**: Prints the value of the epsilon constant used in TensorFlow/Keras operations. This block runs only if the script is executed as the main program.

### Summary

The `model.py` script defines a custom model architecture (`AS_Net`) that leverages different encoder architectures (VGG16, MobileNetV3Large, EfficientNetV2L) and custom attention mechanisms (SAM and CAM). It includes:

- **Feature Extraction**: Uses pre-trained encoders to extract feature maps from specific layers.
- **Feature Map Adjustment**: Adjusts feature map sizes to ensure compatibility for concatenation.
- **Attention Modules**: Applies custom SAM and CAM modules to refine the feature maps.
- **Model Construction**: Constructs the final model with a series of convolutional, pooling, and dense layers.

The `SAM`, `CAM`, and `Synergy` classes define specialized modules used within the `AS_Net` model for enhancing feature representations through attention mechanisms and feature combination strategies.

<!-- 6 -->

## evaluate

[↑](#contents)

### Import Statements

1. **`from sklearn.metrics import classification_report, confusion_matrix`**:

   - **Purpose**: Imports metrics for evaluating classification performance.

2. **`import matplotlib.pyplot as plt`**:

   - **Purpose**: Imports Matplotlib for plotting results.

3. **`import numpy as np`**:

   - **Purpose**: Imports NumPy for numerical operations and array handling.

4. **`from model import AS_Net`**:

   - **Purpose**: Imports the `AS_Net` model definition from the `model.py` script.

5. **`from config import chosen_encoder`**:

   - **Purpose**: Imports the `chosen_encoder` configuration, which indicates which encoder was used during training.

6. **`import os`**:
   - **Purpose**: Imports OS for file and directory operations.

### Environment and Configuration

1. **`os.environ["CUDA_VISIBLE_DEVICES"] = "0"`**:

   - **Purpose**: Ensures that only the first GPU is visible to TensorFlow, useful for managing GPU resources.

2. **`np.set_printoptions(threshold=np.inf)`**:
   - **Purpose**: Configures NumPy to print full arrays without truncation, which is useful for debugging.

### Load Data

1. **`val_data = np.load('data/data_val.npy').astype(np.float32)`**:

   - **Purpose**: Loads the validation data and converts it to `float32` type.

2. **`val_labels = np.load('data/label_val.npy').astype(np.int32)`**:

   - **Purpose**: Loads the validation labels and converts them to `int32` type.

3. **`label_classes = np.load('data/label_classes.npy', allow_pickle=True)`**:
   - **Purpose**: Loads the class names for the labels, allowing for label class names to be stored in a pickle format.

### Data Normalization

1. **`val_data = tf.image.adjust_gamma(val_data / 255., gamma=1.6)`**:
   - **Purpose**: Normalizes the validation data by adjusting the gamma of the pixel values. Pixel values are scaled to the range [0, 1] and gamma correction is applied.

### Initialize Model

1. **`model = AS_Net(encoder=chosen_encoder)`**:

   - **Purpose**: Initializes the model using the encoder specified in the configuration.

2. **`weights_path = f'./checkpoint/{chosen_encoder}_weights.weights.h5'`**:

   - **Purpose**: Constructs the path to the saved model weights based on the chosen encoder.

3. **`if os.path.exists(weights_path):`**:
   - **Purpose**: Checks if the weight file exists. If it does, it loads the weights into the model; otherwise, it prints an error and exits.

### Predict

1. **`predictions = model.predict(val_data, batch_size=1, verbose=1)`**:

   - **Purpose**: Uses the model to make predictions on the validation data. Predictions are made one batch at a time.

2. **`unique_predictions = np.unique(np.argmax(predictions, axis=-1))`**:

   - **Purpose**: Finds unique classes predicted by the model, useful for debugging.

3. **`if len(unique_predictions) == 1:`**:
   - **Purpose**: Checks if the model is predicting only one class. If so, prints a warning.

### Evaluation Metrics

1. **`y_pred = np.argmax(predictions, axis=-1)`**:

   - **Purpose**: Converts the model predictions from probabilities to class labels.

2. **`y_true = val_labels`**:

   - **Purpose**: Sets the true labels for validation data.

3. **`output_folder = 'output/'`**:

   - **Purpose**: Defines the directory where evaluation results will be saved.

4. **`os.makedirs(output_folder, exist_ok=True)`**:

   - **Purpose**: Creates the output directory if it doesn’t already exist.

5. **`print("\nClassification Report:")`**:

   - **Purpose**: Prints a classification report which includes precision, recall, f1-score, and support for each class.

6. **`cm = confusion_matrix(y_true, y_pred)`**:

   - **Purpose**: Computes the confusion matrix to evaluate the performance of the classification model.

7. **`plt.figure(figsize=(10, 8))`**:

   - **Purpose**: Sets up the figure size for the confusion matrix plot.

8. **`plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)`**:

   - **Purpose**: Displays the confusion matrix as an image with a color map.

9. **`plt.savefig(output_folder + f'{chosen_encoder}_confusion_matrix.png')`**:
   - **Purpose**: Saves the confusion matrix plot to a file.

### Save Evaluation Results

1. **`with open(output_folder + f'{chosen_encoder}_performances.txt', 'w') as file_perf:`**:

   - **Purpose**: Opens a file to save the classification report and confusion matrix.

2. **`file_perf.write(...)`**:
   - **Purpose**: Writes the classification report and confusion matrix to the file.

### Save Sample Results

1. **`fig, ax = plt.subplots(8, 2, figsize=[15, 48])`**:

   - **Purpose**: Creates a subplot to display sample images.

2. **`for idx in range(min(8, len(val_data))):`**:

   - **Purpose**: Iterates through the first 8 samples to display true and predicted labels.

3. **`plt.savefig(output_folder + f'{chosen_encoder}_sample_results.png')`**:
   - **Purpose**: Saves the sample results plot to a file.

### Conclusion

1. **`print("Evaluation completed. Results saved in the 'output' folder.")`**:
   - **Purpose**: Prints a message indicating the completion of the evaluation and where the results are saved.

### Summary

The `evaluate.py` script performs the following tasks:

1. **Loading and Normalizing Data**: Loads and preprocesses validation data.
2. **Initializing the Model**: Sets up the model with the chosen encoder and loads the weights.
3. **Making Predictions**: Uses the model to predict labels for validation data.
4. **Evaluating Performance**: Computes and displays classification metrics, confusion matrix, and visualizations.
5. **Saving Results**: Saves performance metrics and sample results for review.

This script is comprehensive and covers model evaluation from prediction to visualization, ensuring that you can thoroughly assess your model’s performance on the validation dataset.
