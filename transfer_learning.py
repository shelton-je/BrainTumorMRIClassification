import os
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description='transfer learning with augmented data or only reals')


parser.add_argument('--type', type=str, choices=['augment', 'normal', 'generated'],
                    help='Type of operation to perform', required=True)

# Parse the arguments
args = parser.parse_args()

TRAIN_PATH = ''
VALID_PATH = ''

# Use the arguments 
if args.type == 'augment':
    TRAIN_PATH = 'Training_data_with_fakes'
    VALID_PATH = 'Brain_MRI_Images/Validation'
elif args.type == 'normal':
    TRAIN_PATH = 'Brain_MRI_Images/Train'
    VALID_PATH = 'Brain_MRI_Images/Validation'
elif args.type == 'generated':
    TRAIN_PATH = 'Brain_MRI_Images/Train'
    VALID_PATH = 'Brain_MRI_Images/Validation'

CATEGORIES = {
    'Tumor': '1',
    'Normal': '0'
}

# Function to generate data frame for the file paths and labels
def generate_df(directory: str, categories: dict) -> pd.DataFrame:
    data = []

    for category, label in categories.items():
        category_path = os.path.join(directory, category)
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            data.append([file_path, label])

    return pd.DataFrame(data, columns=['path', 'label'])

# Generate data frames
train_df = generate_df(TRAIN_PATH, CATEGORIES)
valid_df = generate_df(VALID_PATH, CATEGORIES)

# Data generator for normalization and augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
)

# Data generators for training and validation
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="path",
    y_col="label",
    target_size=(224, 224),  # Resizing to match VGG16 input
    batch_size=32,
    class_mode='binary',
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=None,
    x_col="path",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create the model and add custom layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)