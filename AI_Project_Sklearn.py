#Author: Chi Nguyen, Sammy Cui
#Purpose: AI project 2, classifying cats and dogs


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


IMG_SIZE = 50

training_data = []

testing_data = []

DATADIR = os.path.expanduser("~/Documents/cats-dogs/PetImages")

CATEGORIES = ["Dog", "Cat"]
for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    class_num = CATEGORIES.index(category)
    i = 0
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        if i < 100:

            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            i +=1
        elif i >=100 and i < 200:
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                testing_data.append([new_array, class_num])
            except Exception as e:
                pass
            i += 1
        else:
            break




random.shuffle(training_data)
random.shuffle(testing_data)

x = []
y = []

x_test = []
y_test = []

for features,label in testing_data:
    x_test.append(features)
    y_test.append(label)

x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test = x_test.reshape(200, 2500).astype(float)


for features,label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

x = x.reshape(200, 2500).astype(float)

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('select', SelectKBest()),
    ('classify', SVC(kernel='linear'))
])

# Hyper-parameters we want to try
settings = {
    'scale__with_mean': [False, True],
    'scale__with_std': [False, True],
    'select__k': [1, 2, 3, 4],
    'classify__C': [0.1, 1, 10]
}
grid = GridSearchCV(pipe, settings, cv=2)
scores = cross_val_score(grid, x, y, cv=2)
print("Scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

# Print the best settings to use in this pipeline later
grid.fit(x, y)
print("Best settings:", grid.best_params_)

score = grid.score(x_test, y_test)
print(score)
# Get an unbiased estimate of the score for this pipeline

