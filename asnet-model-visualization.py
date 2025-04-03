# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # AS-Net Model Visualization
# This notebook visualizes the AS-Net architecture with different encoders (VGG16, MobileNetV3, EfficientNetV2)

# ## Import Libraries

# %pip install tensorflow keras matplotlib pydot graphviz --quiet


import os
import gc
import tensorflow as tf
from keras import Model, Input, backend
from keras.applications import VGG16
from keras.applications import MobileNetV3Large, MobileNetV3Small
from keras.applications import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2
from keras.layers import (
    Conv2D, UpSampling2D,
    concatenate, Layer
)
import matplotlib.pyplot as plt
# Add this import for rounded rectangles
from matplotlib.patches import FancyBboxPatch
from tensorflow.keras.utils import plot_model

# Set memory growth to prevent GPU memory overflow
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# ## Define Attention Modules (SAM, CAM, Synergy)
# These modules are common across all AS-Net variants

# --- SAM Module ---


class SAM(tf.keras.Model):
    """Spatial Attention Module"""

    def __init__(self, filters, name='sam', **kwargs):
        super(SAM, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.out_channels = filters // 4  # Output channels for SAM/CAM in AS-Net paper

        # Convolution layers
        self.conv1 = tf.keras.layers.Conv2D(
            self.out_channels, 3, activation="relu", padding="same", name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(
            self.out_channels, 3, activation="relu", padding="same", name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(
            self.out_channels, 3, activation="relu", padding="same", name='conv3')
        self.conv4 = tf.keras.layers.Conv2D(
            self.out_channels, 1, activation="relu", name='conv4')

        # Pooling and Upsampling
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')
        self.pool2 = tf.keras.layers.MaxPooling2D((4, 4), name='pool2')

        # We'll use tf.image.resize in the call method instead of fixed UpSampling2D
        # to ensure exact dimensions match
        self.W1 = tf.keras.layers.Conv2D(
            self.out_channels, 1, activation="sigmoid", name='W1')
        self.W2 = tf.keras.layers.Conv2D(
            self.out_channels, 1, activation="sigmoid", name='W2')

        self.activation = tf.keras.layers.Activation('relu', name='relu_act')
        self.multiply = tf.keras.layers.Multiply(name='multiply')
        self.add = tf.keras.layers.Add(name='add')

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))
        out2 = self.conv4(inputs)

        # Attention branches with explicit resize to match input size
        pooled1 = self.pool1(out2)
        # Use tf.image.resize instead of UpSampling2D to ensure exact size
        upsampled1 = tf.image.resize(pooled1, tf.shape(out2)[
                                     1:3], method='bilinear')
        attention1 = self.W1(upsampled1)

        pooled2 = self.pool2(out2)
        upsampled2 = tf.image.resize(pooled2, tf.shape(out2)[
                                     1:3], method='bilinear')
        attention2 = self.W2(upsampled2)

        # Combine attention maps
        attention_combined = self.add([attention1, attention2])

        # Apply attention and add skip connection
        attended_features = self.multiply([out1, attention_combined])
        output = self.add([attended_features, out2])

        return output

# --- CAM Module ---


class CAM(tf.keras.Model):
    """Channel Attention Module"""

    def __init__(self, filters, reduction_ratio=16, name='cam', **kwargs):
        super(CAM, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.out_channels = filters // 4  # Output channels for SAM/CAM in AS-Net paper
        self.reduction_ratio = reduction_ratio

        # Convolution layers
        self.conv1 = tf.keras.layers.Conv2D(
            self.out_channels, 3, activation="relu", padding="same", name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(
            self.out_channels, 3, activation="relu", padding="same", name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(
            self.out_channels, 3, activation="relu", padding="same", name='conv3')
        self.conv4 = tf.keras.layers.Conv2D(
            self.out_channels, 1, activation="relu", name='conv4')

        # Channel attention mechanism
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(
            name='global_avg_pool')
        self.fc1 = tf.keras.layers.Dense(
            self.out_channels // self.reduction_ratio, activation="relu", name='fc1')
        self.fc2 = tf.keras.layers.Dense(
            self.out_channels, activation="sigmoid", name='fc2')
        self.reshape = tf.keras.layers.Reshape(
            (1, 1, self.out_channels), name='reshape')
        self.multiply = tf.keras.layers.Multiply(name='multiply')
        self.add = tf.keras.layers.Add(name='add')

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))
        out2 = self.conv4(inputs)

        # Channel attention calculation
        pooled = self.global_pool(out2)
        fc1_out = self.fc1(pooled)
        fc2_out = self.fc2(fc1_out)
        channel_weights = self.reshape(fc2_out)

        # Apply channel attention and add skip connection
        attended_features = self.multiply([out1, channel_weights])
        output = self.add([attended_features, out2])

        return output

# --- Synergy Module ---


class Synergy(tf.keras.Model):
    """Combines SAM and CAM outputs with learnable weights."""

    def __init__(self, alpha_init=0.5, beta_init=0.5, name='synergy', **kwargs):
        super(Synergy, self).__init__(name=name, **kwargs)
        self.alpha = tf.Variable(
            alpha_init, trainable=True, name="alpha", dtype=tf.float32)
        self.beta = tf.Variable(beta_init, trainable=True,
                                name="beta", dtype=tf.float32)

        # 1x1 Convolution after weighted sum
        self.conv = tf.keras.layers.Conv2D(1, 1, padding="same", name='conv')
        self.bn = tf.keras.layers.BatchNormalization(name='bn')
        self.add = tf.keras.layers.Add(name='add_weighted')

    def call(self, inputs):
        sam_features, cam_features = inputs

        # Cast learnable weights to input dtype
        alpha_casted = tf.cast(self.alpha, sam_features.dtype)
        beta_casted = tf.cast(self.beta, cam_features.dtype)

        # Weighted sum
        weighted_sum = self.add(
            [alpha_casted * sam_features, beta_casted * cam_features])

        # Apply Conv -> BN
        output = self.conv(weighted_sum)
        output = self.bn(output)

        return output

# Custom layer for resizing features


class ResizeToMatchLayer(Layer):
    """Resizes input tensor to match the spatial dimensions of the target tensor."""

    def __init__(self, name=None, **kwargs):
        super(ResizeToMatchLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        # inputs[0] is the tensor to resize, inputs[1] is the target tensor
        x_to_resize, target = inputs
        target_shape = tf.shape(target)
        target_height, target_width = target_shape[1], target_shape[2]
        return tf.image.resize(x_to_resize, [target_height, target_width], method='bilinear')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][1], input_shape[1][2], input_shape[0][3])

# ## VGG16 AS-Net Model Definition


def build_vgg16_asnet(input_size=(192, 192, 3)):
    """Builds AS-Net with VGG16 encoder for visualization."""
    inputs = Input(input_size, dtype=tf.float32, name='input_image')

    # Load VGG16 backbone (weights don't matter for visualization)
    VGGnet = VGG16(weights=None, include_top=False, input_tensor=inputs)

    # Extract feature maps from VGG16 encoder stages
    # block1_conv2, Shape: (H, W, 64)
    output1 = VGGnet.get_layer(index=2).output
    # block2_conv2, Shape: (H/2, W/2, 128)
    output2 = VGGnet.get_layer(index=5).output
    # block3_conv3, Shape: (H/4, W/4, 256)
    output3 = VGGnet.get_layer(index=9).output
    # block4_conv3, Shape: (H/8, W/8, 512)
    output4 = VGGnet.get_layer(index=13).output
    # block5_conv3, Shape: (H/16, W/16, 512)
    output5 = VGGnet.get_layer(index=17).output

    # --- Decoder with SAM, CAM, and Synergy ---
    # Upsample block 5, concatenate with block 4
    up5 = UpSampling2D((2, 2), interpolation="bilinear", name='up5')(output5)
    # Shape: (H/8, W/8, 512+512=1024)
    merge1 = concatenate([output4, up5], axis=-1, name='merge1')

    # Apply SAM and CAM at the first decoder stage
    # Input filters = 1024. Output filters = 1024 // 4 = 256
    SAM1 = SAM(filters=1024, name='sam1')(merge1)
    CAM1 = CAM(filters=1024, name='cam1')(merge1)

    # Upsample SAM1/CAM1, concatenate with block 3
    up_sam1 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam1')(SAM1)
    up_cam1 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam1')(CAM1)
    merge21 = concatenate([output3, up_sam1], axis=-1, name='merge21')
    merge22 = concatenate([output3, up_cam1], axis=-1, name='merge22')

    # Apply SAM and CAM at the second decoder stage
    SAM2 = SAM(filters=512, name='sam2')(merge21)
    CAM2 = CAM(filters=512, name='cam2')(merge22)

    # Upsample SAM2/CAM2, concatenate with block 2
    up_sam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam2')(SAM2)
    up_cam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam2')(CAM2)
    merge31 = concatenate([output2, up_sam2], axis=-1, name='merge31')
    merge32 = concatenate([output2, up_cam2], axis=-1, name='merge32')

    # Apply SAM and CAM at the third decoder stage
    SAM3 = SAM(filters=256, name='sam3')(merge31)
    CAM3 = CAM(filters=256, name='cam3')(merge32)

    # Upsample SAM3/CAM3, concatenate with block 1
    up_sam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam3')(SAM3)
    up_cam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam3')(CAM3)
    merge41 = concatenate([output1, up_sam3], axis=-1, name='merge41')
    merge42 = concatenate([output1, up_cam3], axis=-1, name='merge42')

    # Apply SAM and CAM at the fourth decoder stage
    SAM4 = SAM(filters=128, name='sam4')(merge41)
    CAM4 = CAM(filters=128, name='cam4')(merge42)

    # Synergy module to combine final SAM and CAM outputs
    synergy_output = Synergy(name='synergy')([SAM4, CAM4])

    # Final 1x1 convolution for segmentation map
    output = Conv2D(1, 1, padding="same", activation="sigmoid",
                    name='final_output')(synergy_output)

    # Create the model
    model = Model(inputs=inputs, outputs=output, name='AS_Net_VGG16')
    return model

# ## MobileNetV3 AS-Net Model Definition


def build_mobilenetv3_asnet(input_size=(224, 224, 3), variant='Large'):
    """Builds AS-Net with MobileNetV3 encoder for visualization."""
    inputs = Input(input_size, dtype=tf.float32, name='input_image')

    # Load MobileNetV3 backbone (weights don't matter for visualization)
    if variant == 'Large':
        base_model = MobileNetV3Large(
            weights=None, include_top=False, input_tensor=inputs)
        # Skip layer names for Large variant
        skip_layer_names = [
            're_lu', 'expanded_conv_2_add', 'expanded_conv_5_add',
            'expanded_conv_14_add', 'activation_19'
        ]
    else:  # Small
        base_model = MobileNetV3Small(
            weights=None, include_top=False, input_tensor=inputs)
        # Skip layer names for Small variant
        skip_layer_names = [
            're_lu', 'expanded_conv_1_project_bn', 'expanded_conv_3_project_bn',
            'expanded_conv_7_add', 'activation_17'
        ]

    # Extract feature maps from the encoder
    encoder_outputs = []
    for name in skip_layer_names:
        try:
            layer_output = base_model.get_layer(name).output
            encoder_outputs.append(layer_output)
        except ValueError:
            # For visualization, we don't need exact layer matches
            pass

    # Ensure we have five encoder outputs (or create dummy outputs for visualization)
    while len(encoder_outputs) < 5:
        encoder_outputs.append(
            Conv2D(64, 1, name=f'dummy_encoder_output_{len(encoder_outputs)}')(inputs))

    output1, output2, output3, output4, bottleneck = encoder_outputs

    # Decoder Stage 1: H/32 -> H/16
    # Use ResizeToMatchLayer to ensure spatial dimensions match exactly
    up4 = UpSampling2D((2, 2), interpolation="bilinear",
                       name='up4')(bottleneck)
    # Fix: Use ResizeToMatchLayer to ensure dimensions match before concatenate
    up4 = ResizeToMatchLayer(name='resize_up4')([up4, output4])
    merge4 = concatenate([output4, up4], axis=-1, name='merge4')
    SAM4 = SAM(filters=merge4.shape[-1], name='sam4')(merge4)
    CAM4 = CAM(filters=merge4.shape[-1], name='cam4')(merge4)

    # Decoder Stage 2: H/16 -> H/8
    up_sam4 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam4')(SAM4)
    up_cam4 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam4')(CAM4)
    # Fix: Use ResizeToMatchLayer for all upsampled features
    up_sam4 = ResizeToMatchLayer(
        name='resize_sam4_to_out3')([up_sam4, output3])
    up_cam4 = ResizeToMatchLayer(
        name='resize_cam4_to_out3')([up_cam4, output3])
    merge31 = concatenate([output3, up_sam4], axis=-1, name='merge31')
    merge32 = concatenate([output3, up_cam4], axis=-1, name='merge32')
    SAM3 = SAM(filters=merge31.shape[-1], name='sam3')(merge31)
    CAM3 = CAM(filters=merge32.shape[-1], name='cam3')(merge32)

    # Decoder Stage 3: H/8 -> H/4
    up_sam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam3')(SAM3)
    up_cam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam3')(CAM3)
    # Fix: Use ResizeToMatchLayer
    up_sam3 = ResizeToMatchLayer(
        name='resize_sam3_to_out2')([up_sam3, output2])
    up_cam3 = ResizeToMatchLayer(
        name='resize_cam3_to_out2')([up_cam3, output2])
    merge21 = concatenate([output2, up_sam3], axis=-1, name='merge21')
    merge22 = concatenate([output2, up_cam3], axis=-1, name='merge22')
    SAM2 = SAM(filters=merge21.shape[-1], name='sam2')(merge21)
    CAM2 = CAM(filters=merge22.shape[-1], name='cam2')(merge22)

    # Decoder Stage 4: H/4 -> H/2
    up_sam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam2')(SAM2)
    up_cam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam2')(CAM2)
    # Fix: Use ResizeToMatchLayer
    up_sam2 = ResizeToMatchLayer(
        name='resize_sam2_to_out1')([up_sam2, output1])
    up_cam2 = ResizeToMatchLayer(
        name='resize_cam2_to_out1')([up_cam2, output1])
    merge11 = concatenate([output1, up_sam2], axis=-1, name='merge11')
    merge12 = concatenate([output1, up_cam2], axis=-1, name='merge12')
    SAM1 = SAM(filters=merge11.shape[-1], name='sam1')(merge11)
    CAM1 = CAM(filters=merge12.shape[-1], name='cam1')(merge12)

    # Final Upsampling Stage: H/2 -> H (Full Resolution)
    final_up_sam = UpSampling2D(
        (2, 2), interpolation="bilinear", name='final_up_sam')(SAM1)
    final_up_cam = UpSampling2D(
        (2, 2), interpolation="bilinear", name='final_up_cam')(CAM1)

    # Synergy module to combine final SAM and CAM outputs
    synergy_output = Synergy(name='synergy')([final_up_sam, final_up_cam])

    # Final 1x1 convolution for segmentation map
    output = Conv2D(1, 1, padding="same", activation="sigmoid",
                    name='final_output')(synergy_output)

    # Create the model
    model = Model(inputs=inputs, outputs=output,
                  name=f'AS_Net_MobileNetV3_{variant}')
    return model

# ## EfficientNetV2 AS-Net Model Definition


def build_efficientnetv2_asnet(input_size=(240, 240, 3), variant='EfficientNetV2B1'):
    """Builds AS-Net with EfficientNetV2 encoder for visualization."""
    inputs = Input(input_size, dtype=tf.float32, name='input_image')

    # Load EfficientNetV2 backbone (weights don't matter for visualization)
    if variant == 'EfficientNetV2B0':
        base_model = EfficientNetV2B0(
            weights=None, include_top=False, input_tensor=inputs)
        skip_layer_names = [
            'block1a_project_activation', 'block2b_add', 'block3b_add',
            'block5e_add', 'top_activation'
        ]
    elif variant == 'EfficientNetV2B1':
        base_model = EfficientNetV2B1(
            weights=None, include_top=False, input_tensor=inputs)
        skip_layer_names = [
            'block1a_project_activation', 'block2c_add', 'block3c_add',
            'block5f_add', 'top_activation'
        ]
    elif variant == 'EfficientNetV2B2':
        base_model = EfficientNetV2B2(
            weights=None, include_top=False, input_tensor=inputs)
        skip_layer_names = [
            'block1a_project_activation', 'block2c_add', 'block3c_add',
            'block5g_add', 'top_activation'
        ]
    else:
        raise ValueError(f"Unsupported EfficientNetV2 variant: {variant}")

    # Extract feature maps from the encoder
    encoder_outputs = []
    for name in skip_layer_names:
        try:
            layer_output = base_model.get_layer(name).output
            encoder_outputs.append(layer_output)
        except ValueError:
            # For visualization, we don't need exact layer matches
            pass

    # Ensure we have five encoder outputs (or create dummy outputs for visualization)
    while len(encoder_outputs) < 5:
        encoder_outputs.append(
            Conv2D(64, 1, name=f'dummy_encoder_output_{len(encoder_outputs)}')(inputs))

    # Unpack encoder outputs
    output1, output2, output3, output4, bottleneck = encoder_outputs

    # Decoder with ResizeToMatchLayer to handle potential size mismatches
    # Decoder Stage 1: Bottleneck -> H/16
    up4 = UpSampling2D(size=(2, 2), interpolation="bilinear",
                       name='up_bottleneck')(bottleneck)
    up4 = ResizeToMatchLayer(name='resize_up4')([up4, output4])
    merge4 = concatenate([output4, up4], axis=-1, name='merge4')
    SAM4 = SAM(filters=merge4.shape[-1], name='sam4')(merge4)
    CAM4 = CAM(filters=merge4.shape[-1], name='cam4')(merge4)

    # Decoder Stage 2: H/16 -> H/8
    up_sam4 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam4')(SAM4)
    up_cam4 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam4')(CAM4)
    up_sam4 = ResizeToMatchLayer(name='resize_up_sam4')([up_sam4, output3])
    up_cam4 = ResizeToMatchLayer(name='resize_up_cam4')([up_cam4, output3])
    merge31 = concatenate([output3, up_sam4], axis=-1, name='merge31')
    merge32 = concatenate([output3, up_cam4], axis=-1, name='merge32')
    SAM3 = SAM(filters=merge31.shape[-1], name='sam3')(merge31)
    CAM3 = CAM(filters=merge32.shape[-1], name='cam3')(merge32)

    # Decoder Stage 3: H/8 -> H/4
    up_sam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam3')(SAM3)
    up_cam3 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam3')(CAM3)
    up_sam3 = ResizeToMatchLayer(name='resize_up_sam3')([up_sam3, output2])
    up_cam3 = ResizeToMatchLayer(name='resize_up_cam3')([up_cam3, output2])
    merge21 = concatenate([output2, up_sam3], axis=-1, name='merge21')
    merge22 = concatenate([output2, up_cam3], axis=-1, name='merge22')
    SAM2 = SAM(filters=merge21.shape[-1], name='sam2')(merge21)
    CAM2 = CAM(filters=merge22.shape[-1], name='cam2')(merge22)

    # Decoder Stage 4: H/4 -> H/2
    up_sam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_sam2')(SAM2)
    up_cam2 = UpSampling2D(
        (2, 2), interpolation="bilinear", name='up_cam2')(CAM2)
    up_sam2 = ResizeToMatchLayer(name='resize_up_sam2')([up_sam2, output1])
    up_cam2 = ResizeToMatchLayer(name='resize_up_cam2')([up_cam2, output1])
    merge11 = concatenate([output1, up_sam2], axis=-1, name='merge11')
    merge12 = concatenate([output1, up_cam2], axis=-1, name='merge12')
    SAM1 = SAM(filters=merge11.shape[-1], name='sam1')(merge11)
    CAM1 = CAM(filters=merge12.shape[-1], name='cam1')(merge12)

    # Final upsampling
    final_up_sam = UpSampling2D(
        (2, 2), interpolation="bilinear", name='final_up_sam')(SAM1)
    final_up_cam = UpSampling2D(
        (2, 2), interpolation="bilinear", name='final_up_cam')(CAM1)

    # Synergy module
    synergy_output = Synergy(name='synergy')([final_up_sam, final_up_cam])

    # Final output
    output = Conv2D(1, 1, padding="same", activation="sigmoid",
                    name='final_output')(synergy_output)

    # Create the model
    model = Model(inputs=inputs, outputs=output, name=f'AS_Net_{variant}')
    return model

# ## Model Visualization Functions


def visualize_model(model, output_dir='model_visualizations', model_name='model',
                    dpi=150, show_shapes=True, show_dtype=False, show_layer_names=True, rankdir='TB'):
    """Generate and save visualization of model architecture."""
    os.makedirs(output_dir, exist_ok=True)

    # Print model summary
    print(f"\n--- {model_name} Summary ---")
    model.summary()

    # Save model summary to file
    with open(os.path.join(output_dir, f"{model_name}_summary.txt"), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Generate and save model plot with improved settings
    try:
        # Standard plot
        filename = os.path.join(output_dir, f"{model_name}.png")
        plot_model(
            model,
            to_file=filename,
            show_shapes=show_shapes,
            show_dtype=show_dtype,
            show_layer_names=show_layer_names,
            rankdir=rankdir,
            dpi=dpi,
            expand_nested=True,  # Expand nested models
        )
        print(f"Model visualization saved to {filename}")

        # Also generate a horizontal layout version for better visualization
        filename_horizontal = os.path.join(
            output_dir, f"{model_name}_horizontal.png")
        plot_model(
            model,
            to_file=filename_horizontal,
            show_shapes=show_shapes,
            show_dtype=False,  # Reduce clutter
            show_layer_names=show_layer_names,
            rankdir='LR',  # Left to Right layout
            dpi=dpi,
            expand_nested=False  # More compact
        )
        print(f"Horizontal model visualization saved to {filename_horizontal}")
    except Exception as e:
        print(f"Error generating model visualization: {e}")
        print("Try installing graphviz: pip install graphviz pydot")

    # Create a simpler diagram showing the main architecture components
    try:
        create_simplified_diagram(model, output_dir, model_name)
        print(
            f"Simplified diagram saved to {output_dir}/{model_name}_simplified.png")
    except Exception as e:
        print(f"Error creating simplified diagram: {e}")

    # Create detailed model diagram
    try:
        create_detailed_diagram(model, output_dir, model_name)
        print(
            f"Detailed diagram saved to {output_dir}/{model_name}_detailed.png")
    except Exception as e:
        print(f"Error creating detailed diagram: {e}")

# New function to create more appealing, detailed diagrams


def create_detailed_diagram(model, output_dir, model_name):
    """Create a more appealing, detailed diagram of the AS-Net architecture."""
    plt.figure(figsize=(16, 10))

    # Extract encoder type from model name
    encoder_type = model_name.split('_')[-1]
    plt.suptitle(
        f'AS-Net Architecture with {encoder_type} Encoder', fontsize=18)

    # Define the main stages and their positions
    grid_height = 8
    grid_width = 10

    # Create a grid-based diagram with encoder on left, decoder on right

    # Encoder section (left side)
    plt.subplot2grid((grid_height, grid_width), (0, 0),
                     rowspan=grid_height, colspan=3)
    plt.title("Encoder", fontsize=14)

    # Draw encoder stages
    encoder_positions = [(6, 0.5), (5, 0.5), (4, 0.5),
                         (2, 0.5), (0.5, 0.5)]  # From bottom to top
    encoder_labels = ["Input", "Encoder\nBlock 1",
                      "Encoder\nBlock 2", "Encoder\nBlock 3", "Bottleneck"]
    encoder_colors = ['lightblue', 'lightgreen',
                      'lightgreen', 'lightgreen', 'orange']

    for i, (pos, label, color) in enumerate(zip(encoder_positions, encoder_labels, encoder_colors)):
        y, x = pos
        rect_h = 0.8 if i == 0 else 1.0  # Input is smaller
        rect_w = 2.0
        # Use FancyBboxPatch for encoder blocks
        rect = FancyBboxPatch((x - rect_w/2, y - rect_h/2), rect_w, rect_h,
                              boxstyle="round,pad=0.2",
                              facecolor=color,
                              edgecolor='black',
                              alpha=0.8)
        plt.gca().add_patch(rect)
        plt.text(x, y, label, ha='center', va='center',
                 fontsize=10, fontweight='bold')

        # Add downsampling arrows
        if i < len(encoder_positions) - 1:
            y_next, x_next = encoder_positions[i+1]
            plt.arrow(x, y + rect_h/2 + 0.1, 0, y_next - y - rect_h/2 - 0.3,
                      head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Add X marks for feature map output points
    for i in range(1, len(encoder_positions)):
        y, x = encoder_positions[i]
        plt.plot(x + rect_w/2 + 0.1, y, 'x', markersize=8, color='blue')

    plt.axis('off')
    plt.xlim(-0.5, 3)
    plt.ylim(-0.5, 7.5)

    # Decoder section (right side)
    plt.subplot2grid((grid_height, grid_width), (0, 4),
                     rowspan=grid_height, colspan=6)
    plt.title("Decoder with Attention Modules", fontsize=14)

    # Draw decoder stages
    decoder_positions = [(0.5, 0.5), (2, 1.5), (4, 1.5), (6, 1.5)]
    decoder_labels = ["Output\nSegmentation", "Decoder Stage 3\nSAM + CAM",
                      "Decoder Stage 2\nSAM + CAM", "Decoder Stage 1\nSAM + CAM"]

    # Draw skip connections from encoder to decoder
    skip_src_y = [encoder_positions[i][0] for i in range(1, 4)]
    skip_dst_y = [decoder_positions[i][0] for i in range(1, 4)]

    # Draw decoder stages and skip connections
    for i, (pos, label) in enumerate(zip(decoder_positions, decoder_labels)):
        y, x = pos

        if i == 0:  # Output
            rect = plt.Rectangle((x - 0.6, y - 0.4), 1.2, 0.8,
                                 facecolor='lightblue', edgecolor='black', alpha=0.8)
            plt.gca().add_patch(rect)
            plt.text(x, y, label, ha='center', va='center',
                     fontsize=10, fontweight='bold')

            # Arrow from synergy to output
            plt.arrow(x - 1.5, y, 0.5, 0, head_width=0.1,
                      head_length=0.1, fc='black', ec='black')

            # Synergy module
            ellipse = plt.Circle(
                (x - 2, y), 0.5, facecolor='plum', edgecolor='black', alpha=0.8)
            plt.gca().add_patch(ellipse)
            plt.text(x - 2, y, "Synergy\nModule",
                     ha='center', va='center', fontsize=9)

            # Arrows to synergy from SAM/CAM
            plt.arrow(x - 3, y - 0.3, 0.5, 0.3, head_width=0.1,
                      head_length=0.1, fc='black', ec='black')
            plt.arrow(x - 3, y + 0.3, 0.5, -0.3, head_width=0.1,
                      head_length=0.1, fc='black', ec='black')

        else:  # Decoder stages with SAM+CAM
            # SAM box
            sam_rect = FancyBboxPatch((x - 1.5, y - 0.5), 3.0, 0.4,
                                      boxstyle="round,pad=0.2",
                                      facecolor='lightcoral',
                                      edgecolor='black',
                                      alpha=0.8)
            plt.gca().add_patch(sam_rect)
            plt.text(x, y - 0.3, f"SAM {4-i}",
                     ha='center', va='center', fontsize=9)

            # CAM box
            cam_rect = FancyBboxPatch((x - 1.5, y + 0.1), 3.0, 0.4,
                                      boxstyle="round,pad=0.2",
                                      facecolor='lightcoral',
                                      edgecolor='black',
                                      alpha=0.8)
            plt.gca().add_patch(cam_rect)
            plt.text(x, y + 0.3, f"CAM {4-i}",
                     ha='center', va='center', fontsize=9)

            # Feature fusion box
            if i > 0:
                fusion_rect = FancyBboxPatch((x - 0.6, y - 1.2), 1.2, 0.4,
                                             boxstyle="round,pad=0.2",
                                             facecolor='lightyellow',
                                             edgecolor='black',
                                             alpha=0.8)
                plt.gca().add_patch(fusion_rect)
                plt.text(x, y - 1.0, "Feature\nFusion",
                         ha='center', va='center', fontsize=8)

                # Arrows to next stage
                if i < len(decoder_positions) - 1:
                    next_y, next_x = decoder_positions[i+1]

                    # From SAM to next stage
                    plt.arrow(x - 0.5, y - 0.3, -0.5, next_y - y,
                              head_width=0.1, head_length=0.1, fc='black', ec='black')

                    # From CAM to next stage
                    plt.arrow(x - 0.5, y + 0.3, -0.5, next_y - y,
                              head_width=0.1, head_length=0.1, fc='black', ec='black')

            # Draw skip connection from encoder
            if i < 3:  # Skip connections for 3 decoder stages
                skip_idx = 3 - i
                # X position at encoder output
                skip_src_x = encoder_positions[skip_idx][1] + 1.0
                skip_dst_x = x - 2  # X position at decoder input

                plt.arrow(skip_src_x, skip_src_y[i-1], skip_dst_x - skip_src_x, 0,
                          linestyle='--', color='blue', linewidth=1.5,
                          head_width=0.1, head_length=0.1)
                plt.text((skip_src_x + skip_dst_x)/2, skip_src_y[i-1] + 0.2,
                         f"Skip Connection {skip_idx}", color='blue', fontsize=8)

    plt.axis('off')
    plt.xlim(-0.5, 6)
    plt.ylim(-0.5, 7.5)

    # Save the detailed diagram
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(os.path.join(
        output_dir, f"{model_name}_detailed.png"), dpi=200, bbox_inches='tight')
    plt.close()

# ## Visualize AS-Net with Different Encoders


# Create output directory
OUTPUT_DIR = "model_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visualize VGG16 AS-Net
vgg16_asnet = build_vgg16_asnet(input_size=(192, 192, 3))
visualize_model(vgg16_asnet, output_dir=OUTPUT_DIR, model_name='ASNet_VGG16')

# Visualize MobileNetV3 AS-Net (Large variant)
mobilenetv3_large_asnet = build_mobilenetv3_asnet(
    input_size=(224, 224, 3), variant='Large')
visualize_model(mobilenetv3_large_asnet, output_dir=OUTPUT_DIR,
                model_name='ASNet_MobileNetV3_Large')

# Visualize MobileNetV3 AS-Net (Small variant)
mobilenetv3_small_asnet = build_mobilenetv3_asnet(
    input_size=(224, 224, 3), variant='Small')
visualize_model(mobilenetv3_small_asnet, output_dir=OUTPUT_DIR,
                model_name='ASNet_MobileNetV3_Small')

# Visualize EfficientNetV2 AS-Net (B0 variant)
efficientnetv2b0_asnet = build_efficientnetv2_asnet(
    input_size=(224, 224, 3), variant='EfficientNetV2B0')
visualize_model(efficientnetv2b0_asnet, output_dir=OUTPUT_DIR,
                model_name='ASNet_EfficientNetV2B0')

# Visualize EfficientNetV2 AS-Net (B1 variant)
efficientnetv2b1_asnet = build_efficientnetv2_asnet(
    input_size=(240, 240, 3), variant='EfficientNetV2B1')
visualize_model(efficientnetv2b1_asnet, output_dir=OUTPUT_DIR,
                model_name='ASNet_EfficientNetV2B1')

# Optional: Visualize EfficientNetV2 AS-Net (B2 variant) if needed
efficientnetv2b2_asnet = build_efficientnetv2_asnet(
    input_size=(260, 260, 3), variant='EfficientNetV2B2')
visualize_model(efficientnetv2b2_asnet, output_dir=OUTPUT_DIR,
                model_name='ASNet_EfficientNetV2B2')

# ## Generate Additional Architecture Diagrams

# Generate a better diagram showing SAM module structure


def generate_sam_cam_diagrams():
    plt.figure(figsize=(15, 7))

    # SAM Module Diagram (left side)
    plt.subplot(1, 2, 1)
    plt.title('Spatial Attention Module (SAM)', fontsize=16)

    # Create more detailed, appealing diagram with gradients
    # Main paths
    main_path_color = '#add8e6'  # Light blue
    attention_path_color = '#ffcccc'  # Light pink
    shortcut_path_color = '#d3f5d3'  # Light green

    # Define the blocks and their positions (x, y, width, height)
    blocks = [
        # Input block
        ('Input Features', (0.5, 0.9), (0.4, 0.1), 'lightblue'),

        # Main path (left side)
        ('Conv 3×3', (0.3, 0.75), (0.3, 0.08), main_path_color),
        ('Conv 3×3', (0.3, 0.65), (0.3, 0.08), main_path_color),
        ('Conv 3×3', (0.3, 0.55), (0.3, 0.08), main_path_color),

        # Shortcut path (right side)
        ('Conv 1×1', (0.7, 0.75), (0.3, 0.08), shortcut_path_color),

        # Attention mechanisms
        ('MaxPool 2×2', (0.6, 0.45), (0.25, 0.08), attention_path_color),
        ('MaxPool 4×4', (0.8, 0.45), (0.25, 0.08), attention_path_color),
        ('UpSample', (0.6, 0.35), (0.25, 0.08), attention_path_color),
        ('UpSample', (0.8, 0.35), (0.25, 0.08), attention_path_color),
        ('Conv 1×1\nσ', (0.6, 0.25), (0.25, 0.08), attention_path_color),
        ('Conv 1×1\nσ', (0.8, 0.25), (0.25, 0.08), attention_path_color),
        ('Add', (0.7, 0.15), (0.2, 0.08), attention_path_color),

        # Output operations
        ('×', (0.3, 0.3), (0.1, 0.1), 'white'),
        ('+', (0.5, 0.15), (0.1, 0.1), 'white'),
        ('Output', (0.5, 0.05), (0.4, 0.08), 'lightblue'),
    ]

    # Draw blocks with gradient fill and better styling
    for label, (x, y), (w, h), color in blocks:
        if label in ['×', '+']:  # For operation symbols
            circle = plt.Circle((x, y), 0.05, facecolor='white',
                                edgecolor='black', linewidth=1.5)
            plt.gca().add_patch(circle)
            plt.text(x, y, label, ha='center', va='center',
                     fontsize=14, fontweight='bold')
        else:  # For regular blocks
            # Use FancyBboxPatch instead of Rectangle for rounded corners
            rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                                  boxstyle="round,pad=0.3",
                                  facecolor=color,
                                  edgecolor='black',
                                  alpha=0.9,
                                  linewidth=1.5)
            plt.gca().add_patch(rect)
            plt.text(x, y, label, ha='center', va='center', fontsize=10)

    # Add arrows connecting the blocks with better styling
    arrows = [
        # From input to both paths
        ((0.5, 0.85), (0.3, 0.79)),  # Input -> Main path first conv
        ((0.5, 0.85), (0.7, 0.79)),  # Input -> Shortcut path conv

        # Main path flow
        ((0.3, 0.71), (0.3, 0.69)),  # Conv1 -> Conv2
        ((0.3, 0.61), (0.3, 0.59)),  # Conv2 -> Conv3
        ((0.3, 0.51), (0.3, 0.35)),  # Conv3 -> Multiplication point

        # Shortcut flow to attention branches
        ((0.7, 0.71), (0.6, 0.49)),  # Conv1x1 -> MaxPool 2x2
        ((0.7, 0.71), (0.8, 0.49)),  # Conv1x1 -> MaxPool 4x4

        # Attention branches
        ((0.6, 0.41), (0.6, 0.39)),  # MaxPool 2x2 -> UpSample
        ((0.8, 0.41), (0.8, 0.39)),  # MaxPool 4x4 -> UpSample
        ((0.6, 0.31), (0.6, 0.29)),  # UpSample -> Conv1x1 sigmoid
        ((0.8, 0.31), (0.8, 0.29)),  # UpSample -> Conv1x1 sigmoid
        ((0.6, 0.21), (0.65, 0.19)),  # Conv1x1 sigmoid -> Add
        ((0.8, 0.21), (0.75, 0.19)),  # Conv1x1 sigmoid -> Add
        ((0.7, 0.11), (0.35, 0.3)),   # Add -> Multiplication with main path

        # Final operations
        ((0.25, 0.3), (0.45, 0.15)),  # Multiplication output -> Addition
        ((0.7, 0.15), (0.55, 0.15)),  # Shortcut to Addition
        ((0.5, 0.1), (0.5, 0.09)),    # Addition -> Output
    ]

    for (x1, y1), (x2, y2) in arrows:
        plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5,
                                     headwidth=8, headlength=10))

    # Add description text
    plt.text(0.05, 0.02, "SAM applies spatial attention to enhance feature maps.",
             fontsize=10, ha='left', va='bottom')

    # Axis styling
    plt.axis('off')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # CAM Module Diagram (right side) - similar detailed approach
    plt.subplot(1, 2, 2)
    plt.title('Channel Attention Module (CAM)', fontsize=16)

    # Define block positions for CAM
    blocks = [
        # Input block
        ('Input Features', (0.5, 0.9), (0.4, 0.1), 'lightblue'),

        # Main path (left side)
        ('Conv 3×3', (0.3, 0.75), (0.3, 0.08), main_path_color),
        ('Conv 3×3', (0.3, 0.65), (0.3, 0.08), main_path_color),
        ('Conv 3×3', (0.3, 0.55), (0.3, 0.08), main_path_color),

        # Shortcut path (right side)
        ('Conv 1×1', (0.7, 0.75), (0.3, 0.08), shortcut_path_color),

        # Channel attention path
        ('Global Avg\nPooling', (0.7, 0.5), (0.3, 0.08), attention_path_color),
        ('FC Layer\nReLU', (0.7, 0.4), (0.3, 0.08), attention_path_color),
        ('FC Layer\nSigmoid', (0.7, 0.3), (0.3, 0.08), attention_path_color),
        ('Reshape\n(B,1,1,C)', (0.7, 0.2), (0.3, 0.08), attention_path_color),

        # Output operations
        ('×', (0.3, 0.3), (0.1, 0.1), 'white'),
        ('+', (0.5, 0.15), (0.1, 0.1), 'white'),
        ('Output', (0.5, 0.05), (0.4, 0.08), 'lightblue'),
    ]

    # Draw CAM blocks
    for label, (x, y), (w, h), color in blocks:
        if label in ['×', '+']:  # For operation symbols
            circle = plt.Circle((x, y), 0.05, facecolor='white',
                                edgecolor='black', linewidth=1.5)
            plt.gca().add_patch(circle)
            plt.text(x, y, label, ha='center', va='center',
                     fontsize=14, fontweight='bold')
        else:  # For regular blocks
            # Use FancyBboxPatch for rounded corners
            rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                                  boxstyle="round,pad=0.3",
                                  facecolor=color,
                                  edgecolor='black',
                                  alpha=0.9,
                                  linewidth=1.5)
            plt.gca().add_patch(rect)
            plt.text(x, y, label, ha='center', va='center', fontsize=10)

    # Add arrows for CAM
    arrows = [
        # From input to both paths
        ((0.5, 0.85), (0.3, 0.79)),  # Input -> Main path first conv
        ((0.5, 0.85), (0.7, 0.79)),  # Input -> Shortcut path conv

        # Main path flow
        ((0.3, 0.71), (0.3, 0.69)),  # Conv1 -> Conv2
        ((0.3, 0.61), (0.3, 0.59)),  # Conv2 -> Conv3
        ((0.3, 0.51), (0.3, 0.35)),  # Conv3 -> Multiplication point

        # Channel attention flow
        ((0.7, 0.71), (0.7, 0.54)),  # Conv1x1 -> Global Avg Pooling
        ((0.7, 0.46), (0.7, 0.44)),  # Global Avg Pooling -> FC1
        ((0.7, 0.36), (0.7, 0.34)),  # FC1 -> FC2
        ((0.7, 0.26), (0.7, 0.24)),  # FC2 -> Reshape
        ((0.7, 0.16), (0.35, 0.3)),   # Reshape -> Multiplication

        # Final operations
        ((0.25, 0.3), (0.45, 0.15)),  # Multiplication output -> Addition
        ((0.7, 0.15), (0.55, 0.15)),  # Shortcut to Addition
        ((0.5, 0.1), (0.5, 0.09)),    # Addition -> Output
    ]

    for (x1, y1), (x2, y2) in arrows:
        plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5,
                                     headwidth=8, headlength=10))

    # Add description text
    plt.text(0.05, 0.02, "CAM applies channel-wise attention to recalibrate feature channels.",
             fontsize=10, ha='left', va='bottom')

    plt.axis('off')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "SAM_CAM_structure.png"),
                dpi=200, bbox_inches='tight')
    plt.close()

# Generate overall AS-Net architecture diagram


def generate_asnet_overall_diagram():
    plt.figure(figsize=(12, 8))
    plt.title('AS-Net Architecture Overview', fontsize=14)

    # Define main components
    components = [
        ('Input\n(224×224×3)', 0.1, 0.8, 'lightblue'),
        ('Encoder\n(VGG16/MobileNetV3/\nEfficientNetV2)', 0.3, 0.8, 'lightgreen'),

        ('Skip Connection 1', 0.5, 0.9, 'lightyellow'),
        ('Skip Connection 2', 0.5, 0.8, 'lightyellow'),
        ('Skip Connection 3', 0.5, 0.7, 'lightyellow'),
        ('Skip Connection 4', 0.5, 0.6, 'lightyellow'),
        ('Bottleneck', 0.5, 0.5, 'orange'),

        ('Decoder Stage 1\nwith SAM+CAM', 0.7, 0.6, 'lightcoral'),
        ('Decoder Stage 2\nwith SAM+CAM', 0.7, 0.7, 'lightcoral'),
        ('Decoder Stage 3\nwith SAM+CAM', 0.7, 0.8, 'lightcoral'),
        ('Decoder Stage 4\nwith SAM+CAM', 0.7, 0.9, 'lightcoral'),

        ('Synergy\nModule', 0.9, 0.8, 'plum'),
        ('Output\n(224×224×1)', 0.9, 0.6, 'lightblue'),
    ]

    # Draw components
    for name, x, y, color in components:
        rect = plt.Rectangle((x-0.08, y-0.05), 0.16, 0.1,
                             facecolor=color, edgecolor='black')
        plt.gca().add_patch(rect)
        plt.text(x, y, name, ha='center', va='center', fontsize=8, wrap=True)

    # Draw encoder->bottleneck connections
    plt.arrow(0.3, 0.75, 0.12, -0.20, head_width=0.01,
              head_length=0.02, fc='black', ec='black')

    # Draw bottleneck->decoder connection
    plt.arrow(0.57, 0.5, 0.05, 0.10, head_width=0.01,
              head_length=0.02, fc='black', ec='black')

    # Draw skip connections to decoder
    for i, y in enumerate([0.6, 0.7, 0.8, 0.9]):
        # Skip connection -> decoder
        plt.arrow(0.57, y, 0.05, 0, head_width=0.01,
                  head_length=0.02, fc='blue', ec='blue')

    # Draw decoder upsampling path
    for i in range(3):
        y1 = 0.6 + i*0.1
        y2 = 0.7 + i*0.1
        plt.arrow(0.7, y1, 0, 0.05, head_width=0.01,
                  head_length=0.02, fc='black', ec='black')

    # Draw final decoder -> synergy
    plt.arrow(0.78, 0.9, 0.04, -0.05, head_width=0.01,
              head_length=0.02, fc='black', ec='black')
    plt.arrow(0.78, 0.6, 0.04, 0.15, head_width=0.01,
              head_length=0.02, fc='black', ec='black')

    # Draw synergy -> output
    plt.arrow(0.9, 0.75, 0, -0.10, head_width=0.01,
              head_length=0.02, fc='black', ec='black')

    # Add annotations
    plt.text(0.35, 0.30, "AS-Net combines:\n- Skip connections\n- Spatial attention (SAM)\n- Channel attention (CAM)\n- Synergy mechanism",
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'), ha='left', va='top', fontsize=9)

    plt.axis('off')
    plt.xlim(0, 1)
    plt.ylim(0.2, 1)
    plt.savefig(os.path.join(
        OUTPUT_DIR, "ASNet_overall_architecture.png"), dpi=150, bbox_inches='tight')
    plt.close()


# Generate additional diagrams
generate_sam_cam_diagrams()
generate_asnet_overall_diagram()

print(
    f"\nAll model visualizations generated successfully in '{OUTPUT_DIR}' folder")
print("You can now examine the model architectures and diagrams.")

# Clean up resources
backend.clear_session()
gc.collect()
