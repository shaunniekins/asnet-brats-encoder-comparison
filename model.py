# model.py

import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import VGG16, EfficientNetV2L, MobileNetV3Large
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate, Multiply, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

def AS_Net(encoder='vgg16', input_size=(192, 256, 3)):
    inputs = Input(input_size)
    print(f'CURRENT ENCODER: {encoder}')

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

    # Get the output layers dynamically
    output_layers = [ENCODER.get_layer(index=i).output for i in layer_indices]
    outputs = [Model(inputs=ENCODER.inputs, outputs=layer)(inputs) for layer in output_layers]

    # Print shapes for debugging
    for i, output in enumerate(outputs):
        print(f"Output {i+1} shape:", output.shape)

    # Helper function to adjust feature map sizes
    def adjust_feature_map(x, target_shape):
        _, h, w, _ = target_shape
        current_h, current_w = x.shape[1:3]
        if current_h > h or current_w > w:
            return MaxPooling2D(pool_size=(current_h // h, current_w // w))(x)
        elif current_h < h or current_w < w:
            return UpSampling2D(size=(h // current_h, w // current_w))(x)
        return x

    # Adjust and merge feature maps
    merged = outputs[-1]
    for i in range(len(outputs) - 2, -1, -1):
        print(f"Layer {i} output shape: {outputs[i].shape}")
        adjusted = adjust_feature_map(outputs[i], merged.shape)
        merged = concatenate([merged, adjusted], axis=-1)

    # Apply SAM and CAM
    SAM1 = SAM(filters=merged.shape[-1])(merged)
    CAM1 = CAM(filters=merged.shape[-1])(merged)

    # Combine SAM and CAM outputs
    combined = concatenate([SAM1, CAM1], axis=-1)

    # Final layers
    output = Conv2D(64, 3, activation='relu', padding='same')(combined)
    output = GlobalAveragePooling2D()(output)
    output = Dense(4, activation='softmax', kernel_initializer='he_normal')(output)

    model = Model(inputs=inputs, outputs=output)
    return model

class SAM(Model):
    def __init__(self, filters):
        super(SAM, self).__init__()
        self.filters = filters
        self.conv1 = Conv2D(self.filters // 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(self.filters // 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(self.filters // 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv4 = Conv2D(self.filters // 4, 1, activation='relu', kernel_initializer='he_normal')
        self.W1 = Conv2D(self.filters // 4, 1, activation='sigmoid', kernel_initializer='he_normal')
        self.W2 = Conv2D(self.filters // 4, 1, activation='sigmoid', kernel_initializer='he_normal')

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

class CAM(Model):
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

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))
        out2 = self.conv4(inputs)
        out3 = self.fc2(self.fc1(self.gpool(out2)))
        out3 = Reshape((1, 1, self.filters // 4))(out3)
        y = Multiply()([out1, out3]) + out2
        return y

class Synergy(Model):
    def __init__(self, alpha=0.5, beta=0.5):
        super(Synergy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.conv = Conv2D(1, 3, padding='same', kernel_initializer='he_normal')
        self.bn = BatchNormalization()

    def call(self, inputs):
        x, y = inputs
        inputs = self.alpha * x + self.beta * y
        y = self.bn(self.conv(inputs))
        return y

if __name__ == '__main__':
    print(K.epsilon())
