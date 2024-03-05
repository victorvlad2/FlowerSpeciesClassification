import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os


labels_df = pd.read_csv('./flower_labels.csv')

image_folder = './flower_images'

image_size = (128, 128)

def load_images_and_labels(image_folder, labels_df, image_size):
    images = []
    labels = labels_df['label'].values
    for file in labels_df['file']:
        image_path = os.path.join(image_folder, file)
        image = imread(image_path)
        image = resize(image, image_size, anti_aliasing=True)
        images.append(image)
    return np.array(images), labels

images, labels = load_images_and_labels(image_folder, labels_df, image_size)

X = images.reshape(images.shape[0], -1)
y = LabelEncoder().fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")