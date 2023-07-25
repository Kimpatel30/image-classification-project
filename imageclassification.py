import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# Step 1: Dataset Selection
# Example: Using CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Step 2: Data Preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

# Step 3: Pre-trained Model Selection
base_model = tf.keras.applications.VGG16(input_shape=(32, 32, 3), include_top=False, weights='imagenet')

# Step 4: Feature Extraction
train_features = base_model.predict(train_images)
test_features = base_model.predict(test_images)

# Step 5: Model Customization
model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Step 6: Model Training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_features, train_labels, epochs=10, batch_size=128)

# Step 7: Model Evaluation
test_loss, test_acc = model.evaluate(test_features, test_labels)
print("Test Accuracy:", test_acc)
