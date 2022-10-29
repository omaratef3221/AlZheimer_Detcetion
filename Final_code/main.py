import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pandas as pd
import matplotlib.image as img
import json
from sklearn.metrics import classification_report
import cv2
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split as tts
from sklearn import preprocessing
import gc
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
def read_df():
    label = []
    file_path = []
    training_path = 'D:\\Dropbox\\Projects\\Full_Projects\\AlZheimer_Detection\\archive\\AugmentedAlzheimerDataset'
    categories = os.listdir(training_path)
    for category in categories:
        for file in os.listdir(os.path.join(training_path, category)):
            file_path.append(os.path.join(training_path, category, file))
            label.append(category)
    final_data = pd.DataFrame(list(zip(file_path, label)), columns = ["file_path", "label"])
    le = preprocessing.LabelEncoder()
    le.fit(final_data["label"])
    final_data["label"] = le.transform(final_data["label"])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    return final_data

def read_images(final_data):
    print("Reading Images ....")
    images = []
    for i in final_data.iterrows():
        image = cv2.imread(i[1]["file_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150,150))
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        images.append(image)
    images = np.array(images, dtype = "float32")
    return images

def prepare_data(images, final_data):
    label = final_data["label"]
    x_train, x_test, y_train, y_test = tts(images, label, test_size = 0.3) #Train test split
    x_train, x_val, y_train, y_val = tts(x_train, y_train, test_size = 0.15) #Train validation Split
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    print("Training Data Shape is: ", str(x_train.shape), str(y_train.shape))
    print("Validation Data Shape is: ", str(x_val.shape), str(y_val.shape))
    print("Testing Data Shape is: ", str(x_test.shape), str(y_test.shape))
    return x_train, x_val, x_test, y_train, y_val, y_test

def get_model_plot(model):
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

def get_base_model(model_name):
    if model_name == "VGG16":
        return VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
    elif model_name == "VGG19":
        return VGG19(weights="imagenet", include_top=False, input_shape=(150,150,3))
    elif model_name == "EfficientNetB7":
        return EfficientNetB7(weights="imagenet", include_top=False, input_shape=(150,150,3))
    elif model_name == "ResNet50":
        return ResNet50(weights="imagenet", include_top=False, input_shape=(150,150,3))

def build_model(model_name = ""):
    if model_name == "":
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(3,3), input_shape = (150,150,3), activation = "relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(64, kernel_size=(3,3), activation = "relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(32, kernel_size=(3,3), activation = "relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Flatten())
        model.add(Dense(256, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation = "softmax"))
    else:
        base_model = get_base_model(model_name)
        base_model.trainable = False
        model = Sequential([
            base_model,
            Flatten(),
            Dense(4096, activation = "relu"),
            Dropout(0.5),
            Dense(2048, activation = "relu"),
            Dropout(0.5),
            Dense(4, activation = "softmax")
            ])
    print(model.summary())
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.1,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)
    model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),
              metrics = ["accuracy"])
    return model

def train_evaluate_test(model, x_train, x_val, x_test, y_train, y_val, y_test, model_name = "Basic Model"):

    history = model.fit(x_train,y_train,
          validation_data = [x_val, y_val],
          batch_size = 4,
          epochs = 50,
          verbose = 2)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(50)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy for ' + model_name)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss for ' + model_name)
    plt.show()
    print("====================================")
    print("Predictions for model " + model_name)
    print("====================================")

    predictions = model.predict(x_test)
    print(classification_report(y_test, np.round(np.argmax(predictions, axis = 1))))
    gc.collect()
    gc.collect()
