import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
import logging
import datetime
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_file = f'logs/deep_learning_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def show_image(image):
    plt.imshow(image, cmap='Greys')
    plt.savefig(f'logs/sample_image_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def plot_history(history, metrics, model_name):
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        plt.plot(history.history[metric], '-o', label=metric)
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'logs/{model_name}_history_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def predict_image(model, image, model_name):
    show_image(np.reshape(image, [28, 28]))
    pred = model.predict(np.reshape(image, [1, 28, 28, 1]))
    logging.info(f"{model_name} prediction: {pred.argmax()} with confidence: {pred.max():.4f}")

def main():
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting Deep Learning model training")
    logging.info(f"TensorFlow version: {tf.__version__}")

    # Load MNIST dataset
    logging.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    logging.info(f"Training data shape: {x_train.shape}, {y_train.shape}")
    logging.info(f"Test data shape: {x_test.shape}, {y_test.shape}")

    # Display label distribution
    label_counts = collections.Counter(y_train)
    logging.info(f"Label distribution: {sorted(label_counts.items(), key=lambda x: x[0])}")

    # Display sample image
    image_index = 12
    show_image(x_train[image_index])
    logging.info(f"Sample image label: {y_train[image_index]}")

    # Input preprocessing
    logging.info("Preprocessing input data...")
    x_train = np.expand_dims(x_train, -1).astype('float32')
    x_test = np.expand_dims(x_test, -1).astype('float32')
    logging.info(f"Reshaped data: {x_train.shape}, {x_test.shape}")

    # Normalization
    logging.info(f"Before normalization - Max: {x_train.max()}, Min: {x_train.min()}")
    x_train /= x_train.max()
    x_test /= x_train.max()
    logging.info(f"After normalization - Max: {x_train.max()}, Min: {x_train.min()}")

    # Build Vanilla Deep Neural Network
    logging.info("\nBuilding Vanilla DNN model...")
    model = Sequential()
    model.add(tf.keras.Input(shape=(28, 28, 1, )))
    model.add(Flatten())
    model.add(Dense(512, 'relu'))
    model.add(Dense(256, 'sigmoid'))
    model.add(Dense(10, 'softmax'))

    model.summary(print_fn=logging.info)

    # Compile model
    logging.info("Compiling DNN model...")
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    # Setup callbacks
    dnn_log_dir = f"logs/fit/dnn_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    callbacks = [
        ModelCheckpoint(
            'best_dnn_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(log_dir=dnn_log_dir, histogram_freq=1)
    ]

    # Train model
    logging.info("Training DNN model...")
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=256,
        validation_data=(x_test, y_test),
        epochs=10,
        callbacks=callbacks
    )

    # Log DNN training history
    logging.info("\nDNN Training History:")
    for epoch in range(len(history.history['loss'])):
        logging.info(f"Epoch {epoch+1}/10:")
        logging.info(f"  Training Loss: {history.history['loss'][epoch]:.4f}")
        logging.info(f"  Training Accuracy: {history.history['accuracy'][epoch]:.4f}")
        logging.info(f"  Validation Loss: {history.history['val_loss'][epoch]:.4f}")
        logging.info(f"  Validation Accuracy: {history.history['val_accuracy'][epoch]:.4f}")

    # Plot DNN training history
    plot_history(history, metrics=['loss', 'val_loss'], model_name='DNN')
    plot_history(history, metrics=['accuracy', 'val_accuracy'], model_name='DNN')

    # Build CNN
    logging.info("\nBuilding CNN model...")
    cnn = Sequential(name='cnn')
    cnn.add(tf.keras.Input(shape=(28, 28, 1, )))
    cnn.add(Conv2D(5, kernel_size=(3, 3)))
    cnn.add(Flatten())
    cnn.add(Dense(128, 'relu'))
    cnn.add(Dense(10, 'softmax'))

    cnn.summary(print_fn=logging.info)

    # Compile CNN
    logging.info("Compiling CNN model...")
    cnn.compile(
        optimizer='adam',
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    # Setup CNN callbacks
    cnn_log_dir = f"logs/fit/cnn_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    cnn_callbacks = [
        ModelCheckpoint(
            'best_cnn_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(log_dir=cnn_log_dir, histogram_freq=1)
    ]

    # Train CNN
    logging.info("Training CNN model...")
    history = cnn.fit(
        x=x_train,
        y=y_train,
        batch_size=256,
        validation_data=(x_test, y_test),
        epochs=5,
        callbacks=cnn_callbacks
    )

    # Log CNN training history
    logging.info("\nCNN Training History:")
    for epoch in range(len(history.history['loss'])):
        logging.info(f"Epoch {epoch+1}/5:")
        logging.info(f"  Training Loss: {history.history['loss'][epoch]:.4f}")
        logging.info(f"  Training Accuracy: {history.history['accuracy'][epoch]:.4f}")
        logging.info(f"  Validation Loss: {history.history['val_loss'][epoch]:.4f}")
        logging.info(f"  Validation Accuracy: {history.history['val_accuracy'][epoch]:.4f}")

    # Plot CNN training history
    plot_history(history, metrics=['loss', 'val_loss'], model_name='CNN')
    plot_history(history, metrics=['accuracy', 'val_accuracy'], model_name='CNN')

    # Test predictions
    logging.info("\nTesting model predictions...")
    test_image_index = 209
    test_image = x_test[test_image_index]
    predict_image(model, test_image, "DNN")
    predict_image(cnn, test_image, "CNN")

    logging.info(f"\nTraining completed. Log file saved at: {log_file}")
    logging.info("You can view training metrics using TensorBoard with: tensorboard --logdir logs/fit")
    logging.info("\nDone!")

if __name__ == "__main__":
    main() 