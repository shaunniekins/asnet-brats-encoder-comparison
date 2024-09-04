from prepare import train_df, test_df, preprocess_data, create_generators, plot_class_distribution
from model import AS_Net
from training import train_model, plot_training_metrics
from testing import evaluate_model, predict


def main():
    # Load and preprocess data
    tr_df = train_df('brain_tumor_mri/Training')
    ts_df = test_df('brain_tumor_mri/Testing')

    # Plot data distributions
    plot_class_distribution(
        tr_df, title='Count of images in each class (Training)')
    plot_class_distribution(
        ts_df, title='Count of images in each class (Testing)')

    # Preprocess and create data generators
    tr_df, valid_df, ts_df = preprocess_data(tr_df, ts_df)
    tr_gen, valid_gen, ts_gen = create_generators(tr_df, valid_df, ts_df)

    # Instantiate and train models
    encoders = ['mobilenetv3', 'vgg16', 'efficientnetv2']
    for encoder in encoders:
        model = AS_Net(encoder=encoder)
        history = train_model(model, tr_gen, valid_gen)
        plot_training_metrics(history)

        # Evaluate the model
        evaluate_model(model, tr_gen, valid_gen, ts_gen)

        # Make predictions
        predict(model, 'brain_tumor_mri/Testing/meningioma/Te-meTr_0000.jpg',
                tr_gen.class_indices)
        predict(model, 'brain_tumor_mri/Testing/glioma/Te-glTr_0007.jpg',
                tr_gen.class_indices)
        predict(model, 'brain_tumor_mri/Testing/notumor/Te-noTr_0001.jpg',
                tr_gen.class_indices)
        predict(model, 'brain_tumor_mri/Testing/pituitary/Te-piTr_0001.jpg',
                tr_gen.class_indices)


if __name__ == "__main__":
    main()
