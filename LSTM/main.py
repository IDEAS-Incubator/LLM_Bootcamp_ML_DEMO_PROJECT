import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

"""
An LSTM model (Long Short-Term Memory model) is a type of Recurrent Neural Network (RNN) 
designed to learn from sequential data, such as text, time series, or speech. 
LSTMs are especially good at capturing long-range dependencies and patterns in sequences, 
which standard RNNs often struggle with due to the vanishing gradient problem.
"""
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Download IMDB dataset from TensorFlow datasets
print("Loading dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Get word index mapping
word_index = tf.keras.datasets.imdb.get_word_index()

# Create a reverse mapping to decode the reviews
reverse_word_index = {value: key for key, value in word_index.items()}
decode_review = lambda review: ' '.join([reverse_word_index.get(i - 3, '?') for i in review])

# Print sample review
print("\nSample review:")
print(decode_review(x_train[0]))
print(f"Label: {'Positive' if y_train[0] == 1 else 'Negative'}")

# Parameters
vocab_size = 10000
max_length = 250
embedding_dim = 128
lstm_units = 64
batch_size = 64
epochs = 5

# Pad sequences to ensure consistent input shape
print("\nPreparing data...")
x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

# Print data shapes
print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Create validation split from training data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
print(f"Validation data shape: {x_val.shape}")

# Build the LSTM model
print("\nBuilding LSTM model...")
"""
Embedding Layer: Converts word indices to dense vectors.
Bidirectional LSTM Layers: Two stacked LSTM layers (the first returns sequences, the second does not), allowing the model to learn from both past and future context.
Dense Layer: With ReLU activation for further feature extraction.
Dropout Layer: Prevents overfitting.
Output Layer: Single neuron with sigmoid activation for binary classification.
"""
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    Bidirectional(LSTM(lstm_units//2, dropout=0.2, recurrent_dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# Print training history
print("\nTraining History:")
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"  Training Loss: {history.history['loss'][epoch]:.4f}")
    print(f"  Training Accuracy: {history.history['accuracy'][epoch]:.4f}")
    print(f"  Validation Loss: {history.history['val_loss'][epoch]:.4f}")
    print(f"  Validation Accuracy: {history.history['val_accuracy'][epoch]:.4f}")

# Evaluate on test set
print("\n" + "="*50)
print("TESTING ON TEST SET")
print("="*50)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Generate predictions on test set
print("\nGenerating predictions on test set...")
y_pred_prob = model.predict(x_test, verbose=1)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate detailed metrics on test set
print("\nTest Set Classification Report:")
report = classification_report(y_test, y_pred)
print(report)

# Print confusion matrix for test set
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Test Set):")
print(f"                | Predicted Negative | Predicted Positive |")
print(f"Actual Negative | {cm[0][0]:<18} | {cm[0][1]:<18} |")
print(f"Actual Positive | {cm[1][0]:<18} | {cm[1][1]:<18} |")

# Display sample test set predictions
print("\nSample Test Set Predictions:")
for i in range(5):
    review = decode_review(x_test[i])
    prediction = "Positive" if y_pred[i] == 1 else "Negative"
    actual = "Positive" if y_test[i] == 1 else "Negative"
    confidence = y_pred_prob[i][0]
    print("-" * 80)
    print(f"Review: {review[:100]}...")
    print(f"Predicted: {prediction} (confidence: {confidence:.4f}), Actual: {actual}")

# Calculate additional metrics
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
precision = report_dict['weighted avg']['precision']
recall = report_dict['weighted avg']['recall']
f1 = report_dict['weighted avg']['f1-score']

# Summary of test results
print("\n" + "="*50)
print("TEST SET SUMMARY")
print("="*50)
print(f"Total test samples: {len(y_test)}")
print(f"Correct predictions: {sum(y_pred == y_test)}")
print(f"Incorrect predictions: {sum(y_pred != y_test)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Class-specific metrics
print("\nClass-specific metrics:")
print("Negative reviews:")
print(f"  Precision: {report_dict['0']['precision']:.4f}")
print(f"  Recall: {report_dict['0']['recall']:.4f}")
print(f"  F1-Score: {report_dict['0']['f1-score']:.4f}")

print("Positive reviews:")
print(f"  Precision: {report_dict['1']['precision']:.4f}")
print(f"  Recall: {report_dict['1']['recall']:.4f}")
print(f"  F1-Score: {report_dict['1']['f1-score']:.4f}")

# Compare with simple RNN performance
print("\n" + "="*50)
print("LSTM vs RNN COMPARISON")
print("="*50)
print("The LSTM model typically outperforms simple RNN models because:")
print("1. LSTM can capture long-range dependencies better")
print("2. LSTM doesn't suffer from vanishing gradient problem as much")
print("3. The bidirectional architecture captures context from both directions")
print("4. Adding recurrent dropout helps prevent overfitting specific to sequential data")

print("\nDone!")