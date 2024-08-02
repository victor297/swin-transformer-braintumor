import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.optimizers import Adam

# Load and preprocess the data
def load_data(data_dir):
    images = []
    labels = []
    for label in ['yes', 'no']:
        class_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img = img / 255.0  # normalize the image
            images.append(img)
            labels.append(1 if label == 'yes' else 0)
    return np.array(images), np.array(labels)

data_dir = 'dataset'
X, y = load_data(data_dir)

# Reshape the data for LSTM layer
X = X.reshape((X.shape[0], -1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Create the model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
# Save the model
model.save('brain_tumor_detection_model.h5')

print(f'Test accuracy: {accuracy}')
