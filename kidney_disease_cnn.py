import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# --------------------------------------------------
# 1. Dataset paths
# --------------------------------------------------

TRAIN_DIR = "./dataset/train"
VALID_DIR = "./dataset/val"
TEST_DIR = "./dataset/test"


# --------------------------------------------------
# 2. Image preprocessing
# --------------------------------------------------

train_datagen = ImageDataGenerator(rescale=1 / 255)
valid_datagen = ImageDataGenerator(rescale=1 / 255)
test_datagen = ImageDataGenerator(rescale=1 / 255)


train_dataset = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(200, 200),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=100
)

valid_dataset = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(200, 200),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=100
)

test_dataset = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(200, 200),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=100,
    shuffle=False
)


# --------------------------------------------------
# 3. Display sample images
# --------------------------------------------------

def show_image_samples(generator):
    classes = list(generator.class_indices.keys())
    images, labels = next(generator)

    plt.figure(figsize=(12, 12))

    for i in range(min(25, len(labels))):
        plt.subplot(5, 5, i + 1)

        plt.imshow(images[i].squeeze(), cmap="gray")

        class_index = np.argmax(labels[i])
        plt.title(classes[class_index])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


show_image_samples(train_dataset)


# --------------------------------------------------
# 4. Build CNN model
# --------------------------------------------------

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=train_dataset.image_shape
    )
)
model.add(MaxPool2D(2))

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu"
    )
)
model.add(MaxPool2D(2))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    )
)
model.add(MaxPool2D(2))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    )
)
model.add(MaxPool2D(2))

model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    )
)
model.add(MaxPool2D(2))

model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    )
)
model.add(MaxPool2D(2))

model.add(Flatten())

model.add(
    Dense(
        512,
        activation="relu"
    )
)

model.add(
    Dense(
        4,
        activation="softmax"
    )
)


# --------------------------------------------------
# 5. Compile model
# --------------------------------------------------

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------------
# 6. Train model
# --------------------------------------------------

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=5
)


# --------------------------------------------------
# 7. Plot training results
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.show()


# --------------------------------------------------
# 8. Evaluate model
# --------------------------------------------------

predictions = model.predict(test_dataset)

predicted_classes = np.argmax(predictions, axis=1)
actual_classes = test_dataset.classes

accuracy = np.mean(predicted_classes == actual_classes)

precision, recall, f1_score, _ = (
    precision_recall_fscore_support(
        actual_classes,
        predicted_classes,
        average="macro"
    )
)

print("Test Accuracy :", accuracy)
print("Precision      :", precision)
print("Recall         :", recall)
print("F1 Score       :", f1_score)


# --------------------------------------------------
# 9. Confusion Matrix
# --------------------------------------------------

class_names = list(train_dataset.class_indices.keys())

cm = confusion_matrix(
    actual_classes,
    predicted_classes
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Kidney Disease Classification - Confusion Matrix")

plt.tight_layout()
plt.show()
