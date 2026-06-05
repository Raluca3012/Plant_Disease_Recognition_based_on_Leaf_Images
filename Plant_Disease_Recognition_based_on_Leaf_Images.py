import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, DenseNet201
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import *
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

plt.close('all')

path = "C:/Users/Desktop/project/theSmallDataset"

def loadingData(DSpath, imagesize =(224, 224)):
    imgs, lbls = [], []
    for cls in sorted(os.listdir(DSpath)):
        clspath = os.path.join(DSpath, cls)
        if not os.path.isdir(clspath): 
            continue
        for imgnm in os.listdir(clspath):
            img = cv2.imread(os.path.join(clspath, imgnm))
            if img is not None:
                imgs.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), imagesize))
                lbls.append(cls)
    return np.array(imgs), np.array(lbls)

train_path = "C:/Users/rdoro/Desktop/Anul_IV/NNGA_Pr/project/theSmallDataset/train"
valid_path = "C:/Users/rdoro/Desktop/Anul_IV/NNGA_Pr/project/theSmallDataset/valid"
test_path = "C:/Users/rdoro/Desktop/Anul_IV/NNGA_Pr/project/theSmallDataset/test"

def count_images_in_folder(folder_path):
    total = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            total += len(os.listdir(item_path))
        else:
            total += 1
    return total

train_count = count_images_in_folder(train_path)
val_count = count_images_in_folder(valid_path)
test_count = count_images_in_folder(test_path)

print(f"Train images: {train_count}")
print(f"Validation images: {val_count}")
print(f"Test images: {test_count}")

for class_name in os.listdir(train_path):
    class_folder = os.path.join(train_path, class_name)
    files = os.listdir(class_folder)
    print(f"{class_name}: {len(files)} images")

def show_sample_images_with_axes(data_path, classes, image_per_class=1):
    rows, cols = 5, 3
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    selected_classes = classes[:rows * cols]

    for i, class_name in enumerate(selected_classes):
        class_folder = os.path.join(data_path, class_name)
        images = os.listdir(class_folder)[:image_per_class]

        if not images:
            continue

        img_path = os.path.join(class_folder, images[0])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        r = i // cols
        c = i % cols
        ax = axs[r][c]
        ax.imshow(img)
        ax.set_title(class_name, fontsize=10)
        ax.axis('off')

    for i in range(len(selected_classes), rows * cols):
        r = i // cols
        c = i % cols
        axs[r][c].axis('off')

    plt.subplots_adjust(hspace=0.6)
    plt.suptitle("Sample Images from Each Class (Train)", fontsize=16)
    plt.show()

show_sample_images_with_axes(train_path, sorted(os.listdir(train_path)), image_per_class=2)

x_train_origin, y_train = loadingData(train_path)
x_val_origin, y_val = loadingData(valid_path)
x_test_origin, y_test = loadingData(test_path)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)
y_test_encoded = le.transform(y_test)

num_classes = len(np.unique(y_train_encoded))
print(f"{num_classes} unique classes in data.")
print("Classes mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

model_CNN = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  
])

model_CNN.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True
)

datagen_cnn_train = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = datagen_cnn_train.flow(x_train_origin, y_train_encoded, batch_size=64, shuffle=True)
datagen_cnn_val = ImageDataGenerator(rescale=1.0 / 255.0)
val_generator = datagen_cnn_val.flow(x_val_origin, y_val_encoded, batch_size=64, shuffle=False)

history = model_CNN.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop],
    verbose=1
)

model_CNN.save('best_cnn_model.keras')

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_training_history(history)

x_test_norm = x_test_origin / 255.0
y_pred_probs_CNN = model_CNN.predict(x_test_norm)
y_pred_CNN = np.argmax(y_pred_probs_CNN, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print(classification_report(
    y_test_encoded,
    y_pred_CNN,
    target_names=le.classes_
))

cm = confusion_matrix(y_test_encoded, y_pred_CNN)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

accuracy_per_class = []
confidence_per_class = []

for i in range(num_classes):
    idx = np.where(y_test_encoded == i)[0]
    correct = np.sum(y_pred_CNN[idx] == i)
    acc = correct / len(idx) * 100 if len(idx) > 0 else 0
    avg_conf = np.mean([y_pred_probs_CNN[j][y_pred_CNN[j]] for j in idx]) if len(idx) > 0 else 0
    accuracy_per_class.append(acc)
    confidence_per_class.append(avg_conf)

plt.figure(figsize=(10, 5))
sns.barplot(x=accuracy_per_class, y=le.classes_, color='skyblue')
plt.xlabel("Accuracy (%)")
plt.title("Per-Class Accuracy")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=confidence_per_class, y=le.classes_, color='coral')
plt.xlabel("Average Confidence")
plt.title("Per-Class Average Confidence")
plt.tight_layout()
plt.show()

rows, cols = 3, 5
indices = np.random.choice(len(x_test_origin), rows * cols, replace=False)

plt.figure(figsize=(cols * 4, rows * 5))
for i, idx in enumerate(indices):
    img = x_test_origin[idx]
    pred_label = y_pred_CNN[idx]
    true_label = y_test_encoded[idx]
    confidence = y_pred_probs_CNN[idx][pred_label]

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img.astype("uint8"))
    plt.axis("off")
    plt.title(f"Expectation: {le.classes_[true_label]}\n"
              f"Prediction: {le.classes_[pred_label]}\n"
              f"Confidence: {confidence * 100:.1f}%", fontsize=9)

plt.tight_layout()
plt.show()
