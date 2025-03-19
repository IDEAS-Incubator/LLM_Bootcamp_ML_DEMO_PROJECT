import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.datasets import fetch_20newsgroups

# Load a binary classification dataset
categories = ['alt.atheism', 'sci.space']
print("Loading dataset...")
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target, 
    test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Better text preprocessing
print("Preprocessing text data...")
max_words = 10000  # Increased vocabulary size
max_length = 150   # Increased sequence length

# Tokenize the text
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Build a CNN model with slightly more capacity
print("Building CNN model...")
embedding_dim = 100  # Increased embedding dimension

model = Sequential([
    Embedding(max_words, embedding_dim, input_length=max_length),
    Conv1D(128, 5, activation='relu'),  # More filters
    GlobalMaxPooling1D(),
    Dropout(0.4),  # Increased dropout for better regularization
    Dense(64, activation='relu'),  # Added a hidden layer
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile with reduced learning rate for better convergence
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Train the model for more epochs
print("Training model...")
model.fit(
    X_train_pad, y_train,
    epochs=10,  # Increased epochs for better training
    batch_size=32,  # Smaller batch size for better generalization
    verbose=1
)

# Evaluate and predict on test data
print("\n--- Test Data Evaluation ---")
loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions on test data
y_pred_prob = model.predict(X_test_pad, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Print full evaluation metrics
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=categories)
print(report)

# Display some sample predictions
print("\n--- Sample Test Predictions ---")
for i in range(5):
    # Get a sample from test data
    sample_text = X_test[i][:100] + "..."  # Truncate for display
    true_label = categories[y_test[i]]
    pred_label = categories[y_pred[i]]
    confidence = y_pred_prob[i][0] if y_pred[i] == 1 else 1 - y_pred_prob[i][0]
    
    print(f"Sample {i+1}:")
    print(f"Text: {sample_text}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {pred_label} (confidence: {confidence:.2f})")
    print(f"Correct prediction: {y_test[i] == y_pred[i]}")
    print("-" * 50)

