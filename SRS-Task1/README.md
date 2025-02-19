# Speaker Recognition System

## Overview
This project implements a speaker recognition system using deep learning techniques. The system is trained to classify audio samples into predefined speaker categories using MFCC (Mel-Frequency Cepstral Coefficients) as features and an LSTM-based model for classification. Hyperparameter tuning is conducted using Keras Tuner to achieve optimal performance.

## Features
- **Audio Preprocessing**: Extracts MFCC features from `.wav` audio files.
- **Deep Learning Model**: Uses LSTM layers for sequential feature learning.
- **Hyperparameter Tuning**: Implements Keras Tuner for optimal hyperparameter selection.
- **Evaluation Metrics**: Includes accuracy, F1-score, and confusion matrix.
- **Real-time Prediction**: Supports inference on new audio files.

## Dataset
The dataset consists of two main classes:
- **Speaker**: Audio samples of a known speaker.
- **Non-Speaker**: Audio samples from other sources.

The dataset is structured as follows:
```
./dataset/
    ├── speaker/
    │   ├── speaker1.wav
    │   ├── speaker2.wav
    │   └── ...
    ├── non_speaker/
    │   ├── noise1.wav
    │   ├── noise2.wav
    │   └── ...
```

## Installation
Ensure you have Python installed, then install the required dependencies:
```sh
pip install tensorflow keras librosa numpy scikit-learn matplotlib seaborn keras-tuner
```

## Preprocessing
1. **Feature Extraction**: The system extracts MFCC features from each audio file.
2. **Normalization**: Standardizes MFCC features using `StandardScaler`.
3. **Label Encoding**: Converts categorical labels into numerical format.

## Model Architecture
- **LSTM Layer**: Captures sequential dependencies in audio features.
- **Dense Layer**: Processes learned features.
- **Softmax Output Layer**: Predicts probabilities for each class.

### Model Training
The model is trained using the following parameters:
- **Best Hyperparameters Found:**
  - LSTM Units: `128`
  - Dense Units: `32`
  - Learning Rate: `0.0005`
- **Training Process:**
  ```python
  history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=[early_stopping])
  ```

### Evaluation
```python
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

### Confusion Matrix
```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

y_pred = np.argmax(model.predict(X_test), axis=1)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```

## Real-time Prediction
```python
predicted_class = classify_audio("test_audio.wav", model, label_encoder)
print(f"Predicted class: {predicted_class}")
```

## Future Enhancements
- Add more speakers for better generalization.
- Implement real-time speaker verification.
- Use Convolutional Recurrent Neural Networks (CRNN) for improved feature extraction.

## Contributors
- **Your Name** - Developer

## License
This project is licensed under the MIT License.

