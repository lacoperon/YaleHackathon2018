from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import click
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception

# Create top level logger
log = logging.getLogger()
log.setLevel(logging.INFO)

# Add console handler using our custom ColoredFormatter
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

# Global parameters

# The image labels.
LABEL_NAMES = np.array([
    'morisons_pouch',
    'bladder',
    'plax',
    '4ch',
    '2ch',
    'ivc',
    'carotid',
    'lungs',
    'thyroid',
])
NUM_CLASSES = len(LABEL_NAMES)

# The size of the raw ultrasound images.
IMAGE_WIDTH = 436
IMAGE_HEIGHT = 512

# The default image size as required by the inception v1 model
TARGET_IMAGE_WIDTH = TARGET_IMAGE_HEIGHT = \
    inception.inception_v1.default_image_size


@click.group()
def cli():
    pass


def predict():
    pass


@cli.command()
@click.option(
    '--input_file',
    required=True,
    type=click.Path(exists=True, dir_okay=False)
)
@click.option('--batch_size', required=False, type=click.INT, default=36)
@click.option(
    '--export_dir',
    required=True,
    type=click.Path(exists=True, dir_okay=True)
)
def evaluate(input_file, batch_size, export_dir):
    """
    :param input_file: the csv file containing the training set.
    :param batch_size: the batch size used for training
    :param export_dir: the checkpoint directory from which the model should
    be restored.

    Example:
    python hackathon_example.py evaluate
    --input_file=butterfly_mini_dataset/test/test.csv
    --export_dir=models
    """
    (
        test_image_paths,
        test_labels,
    ) = load_data_from_csv(
        input_file
    )

    # Define the data iterator.
    image_path_placeholder = tf.placeholder(tf.string, [None])
    label_placeholder = tf.placeholder(tf.int32, [None])

    test_iterator = create_dataset_iterator(
        image_path_placeholder,
        label_placeholder,
        batch_size,
    )
    next_test_batch = test_iterator.get_next()

    # Load from check-point
    with tf.Session() as session:
        new_saver = tf.train.import_meta_graph(
            os.path.join(export_dir, 'buttefly-model.meta')
        )
        new_saver.restore(session, tf.train.latest_checkpoint(export_dir))
        graph = tf.get_default_graph()
        model_input_image = graph.get_tensor_by_name("model_input_image:0")
        expected_label = graph.get_tensor_by_name("expected_label:0")
        predictions = graph.get_tensor_by_name("predictions:0")
        accuracy_to_value, accuracy_update_op = tf.metrics.accuracy(
            predictions,
            expected_label,
        )

        initialize_iterators(
            session,
            test_iterator,
            image_path_placeholder,
            test_image_paths,
            label_placeholder,
            test_labels,
        )

        session.run(tf.local_variables_initializer())

        while True:
            try:
                # Read the next batch.
                batch_images, batch_labels = session.run(
                    next_test_batch,
                )
                # Train the model.
                session.run(
                    [
                        accuracy_update_op,
                    ],
                    feed_dict={
                        model_input_image: batch_images,
                        expected_label: batch_labels,
                    }
                )
            except tf.errors.OutOfRangeError:
                break


        accuracy = session.run(accuracy_to_value)
        print('test accuracy: {}'.format(accuracy,))


@cli.command()
@click.option(
    '--input_file',
    required=True,
    type=click.Path(exists=True, dir_okay=False)
)
@click.option('--batch_size', required=False, type=click.INT, default=36)
@click.option('--number_of_epochs', required=False, type=click.INT, default=10)
@click.option(
    '--export_dir',
    required=True,
    type=click.Path(exists=False, dir_okay=True)
)
def train(input_file, batch_size, number_of_epochs, export_dir):
    """
    :param input_file: the csv file containing the training set.
    :param batch_size: the batch size used for training
    :param number_of_epochs: the number of times the model will be trained
    on all the dataset.
    :param export_dir: The directory where the model will be saved.

    Example:
    python hackathon_example.py train
    --input_file=butterfly_mini_dataset/training/training.csv
    --export_dir=models
    """

    (
        training_image_paths,
        training_labels,
        validation_image_paths,
        validation_labels,
    ) = load_data_from_csv(
        input_file,
        split=True
    )

    # Define the data iterators.
    image_path_placeholder = tf.placeholder(tf.string, [None])
    label_placeholder = tf.placeholder(tf.int32, [None])

    training_iterator = create_dataset_iterator(
        image_path_placeholder,
        label_placeholder,
        batch_size,
    )
    next_training_batch = training_iterator.get_next()

    validation_iterator = create_dataset_iterator(
        image_path_placeholder,
        label_placeholder,
        batch_size,
    )
    next_validation_batch = validation_iterator.get_next()

    # Define inception v1 and return ops to load pre-trained model (trained on
    # ImageNet).
    (
        model_input_image,
        predictions,
        expected_label,
        load_pre_trained_weights_op,
        train_op,
        metrics_to_values,
        metrics_to_updates,
    ) = initialize_pre_trained_inception_v1()

    # Start the training validation loop.
    with tf.Session() as session:

        session.run([
            tf.local_variables_initializer(),
            tf.global_variables_initializer()
        ])
        load_pre_trained_weights_op(session)

        # Define a ModelSaver
        saver = tf.train.Saver()

        training_accuracy = []
        training_loss = []
        validation_accuracy = []
        validation_loss = []
        best_validation_accuracy = None

        # Running training loop.
        for _ in range(number_of_epochs):
            session.run(tf.local_variables_initializer())

            initialize_iterators(
                session,
                training_iterator,
                image_path_placeholder,
                training_image_paths,
                label_placeholder,
                training_labels,
            )

            while True:
                try:
                    # Read the next batch.
                    batch_images, batch_labels = session.run(
                        next_training_batch,
                    )
                    # Train the model.
                    session.run(
                        [
                            metrics_to_updates,
                            train_op,
                        ],
                        feed_dict={
                            model_input_image: batch_images,
                            expected_label: batch_labels,
                        }
                    )
                except tf.errors.OutOfRangeError:
                    break

            metrics_values = session.run(metrics_to_values)

            accuracy = metrics_values['accuracy']
            mean_loss = metrics_values['mean_loss']
            training_accuracy.append(accuracy)
            training_loss.append(mean_loss)
            print('training accuracy: {}, training mean loss: {}'.format(
                accuracy,
                mean_loss))

            # Running validation loop.
            session.run(tf.local_variables_initializer())

            initialize_iterators(
                session,
                validation_iterator,
                image_path_placeholder,
                validation_image_paths,
                label_placeholder,
                validation_labels,
            )

            while True:
                try:
                    # Read the next batch.
                    batch_images, batch_labels = session.run(
                        next_validation_batch,
                    )
                    # Train the model.
                    session.run(
                        metrics_to_updates,
                        feed_dict={
                            model_input_image: batch_images,
                            expected_label: batch_labels,
                        }
                    )
                except tf.errors.OutOfRangeError:
                    break

            metrics_values = session.run(metrics_to_values)
            accuracy = metrics_values['accuracy']
            mean_loss = metrics_values['mean_loss']
            validation_accuracy.append(accuracy)
            validation_loss.append(mean_loss)
            print('validation accuracy: {}, validation mean loss: {}'.format(
                accuracy,
                mean_loss))
            # Save model if accuracy improved.
            if (
                    (not best_validation_accuracy) or
                    best_validation_accuracy < accuracy
            ):
                best_validation_accuracy = accuracy
                saver.save(session, os.path.join(export_dir, 'buttefly-model'))


def initialize_pre_trained_inception_v1():
    """
    :param session: The current tensorflow session.
    :param model_input_image: A placeholder for the input images.
    :param expected_label: A placeholder for the expected labels.
    :return: A tuple containing the following ops:
            loss, train_op, accuracy, accuracy_update, mean_loss,
            mean_loss_update,
    """

    # Define input and output to the inception v1 model.
    model_input_image = tf.placeholder(
        tf.float32,
        [None, TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT, 3],
        name="model_input_image"
    )
    expected_label = tf.placeholder(
        tf.int64,
        [None],
        name="expected_label"
    )

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        #  Load the deep learning model.
        logits, end_points = inception.inception_v1(
            model_input_image,
            num_classes=NUM_CLASSES,
            is_training=False
        )

        # We are going to train only the last layer of the model.
        trainable_layer = 'InceptionV1/Logits/Conv2d_0c_1x1'

        variables_to_restore = slim.get_variables_to_restore(
            exclude=[trainable_layer]
        )
        variables_to_train = slim.get_variables_by_suffix('',
                                                          trainable_layer)

        # Transform the labels into one hot encoding.
        one_hot_labels = tf.one_hot(
            expected_label,
            NUM_CLASSES,
        )

        # Define the loss function.
        loss = tf.losses.softmax_cross_entropy(
            one_hot_labels,
            end_points['Logits'],
        )

        # Select the optimizer.
        optimizer = tf.train.AdamOptimizer(1e-4)

        # Create a train op.
        train_op = tf.contrib.training.create_train_op(
            loss,
            optimizer,
            variables_to_train=variables_to_train,
        )

        predictions = tf.identity(
            tf.argmax(end_points['Predictions'], 1),
            name="predictions"
        )
        metrics_to_values, metrics_to_updates = \
            slim.metrics.aggregate_metric_map({
                'accuracy': tf.metrics.accuracy(
                    predictions,
                    expected_label,
                ),
                'mean_loss': tf.metrics.mean(loss),
            })

        # Define load predefined model operation.
        load_pre_trained_weights_op = slim.assign_from_checkpoint_fn(
            'inception_v1.ckpt',
            variables_to_restore
        )

        return (
            model_input_image,
            predictions,
            expected_label,
            load_pre_trained_weights_op,
            train_op,
            metrics_to_values,
            metrics_to_updates,
        )


def initialize_iterators(
        session,
        iterator,
        image_path_placeholder,
        image_paths,
        label_placeholder,
        labels
):
    """
    :param session: The current tensorflow session.
    :param iterator: The iterator to initialize.
    :param image_path_placeholder: The image path placeholder.
    :param image_paths: The image paths from which the placeholder will be
    populated.
    :param label_placeholder: The label placeholder.
    :param labels: The labels from which to populate the label placeholder.
    :return:
    """
    session.run(
        iterator.initializer,
        feed_dict={
            image_path_placeholder: image_paths,
            label_placeholder: labels,
        }
    )


def create_dataset_iterator(
        image_placeholder,
        label_placeholder,
        batch_size,
):
    """

    :param image_placeholder: A placeholder for the images.
    :param label_placeholder: A placeholder for the labels.
    :param batch_size: The batch size used by the iterator.
    :return: A tensorflow iterator that can be used to iterate over the
    dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_placeholder, label_placeholder)
    )
    dataset = dataset.map(load_image)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    return dataset.make_initializable_iterator()


def load_data_from_csv(filename, split=False, split_percentage=0.8):
    """
    :param filename: The path to the file to be loaded.
    :param split: whether to split the data into train and validation.
    :param split_percentage: The percentage that will be retained as
    :return: A tuple containing 2 lists in case there is no split: one with
    the image paths and one with the corresponding labels. If split is true
    the method returns 4 lists (2 for training and 2 for validation).
    """
    df = pd.read_csv(filename)
    df = df.sample(frac=1).reset_index(drop=True)
    if split:
        mask = np.random.rand(len(df)) < split_percentage
        train = df[mask]
        validation = df[~mask]
        return (
            train['image_file_path'].tolist(),
            train['label'].tolist(),
            validation['image_file_path'].tolist(),
            validation['label'].tolist(),
        )
    else:
        return (
            df['image_file_path'].tolist(),
            df['label'].tolist(),
        )


def load_image(filepath, label):
    """

    :param filepath: A tensor representing the filepath of the image
    :param label: The label for the image.
    :return: A tensor representing the image ready to be used in the inception
    model and its label.
    """

    image_string = tf.read_file(filepath)
    image_decoded = tf.image.decode_image(
        image_string,
        channels=1
    )
    image_resized = tf.image.resize_image_with_crop_or_pad(
        image_decoded,
        IMAGE_WIDTH,
        IMAGE_HEIGHT
    )
    image_resized = tf.image.resize_images(
        image_resized,
        (
            TARGET_IMAGE_WIDTH,
            TARGET_IMAGE_HEIGHT
        )
    )
    # Normalize the image.
    image_normalized = image_resized / 255
    image = tf.reshape(
        tf.cast(image_normalized, tf.float32),
        shape=(TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT)
    )
    # Stack the image 3 times since the pre-trained inception model
    # required a 3 channel image. This can be optimized by instantiating
    # inception with 1 channel and retrain the first layer from scratch.
    return tf.stack([image, image, image], axis=2), label


# This setup the script so it can be used with different command groups from
# command line.
if __name__ == '__main__':
    cli()
