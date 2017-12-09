import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from sklearn import grid_search
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from lesson_functions import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
#svc = dist_pickle["svc"]
#X_scaler = dist_pickle["scaler"]
#orient = dist_pickle["orient"]
#pix_per_cell = dist_pickle["pix_per_cell"]
#cell_per_block = dist_pickle["cell_per_block"]
#spatial_size = dist_pickle["spatial_size"]
#hist_bins = dist_pickle["hist_bins"]



# Read in car and non-car images

#cars = glob.glob('test_images/vehicles_smallset/*/*.jpeg')
#notcars = glob.glob('test_images/non-vehicles_smallset/*/*.jpeg')
cars = glob.glob('test_images/vehicles/*/*.png')
notcars = glob.glob('test_images/non-vehicles/*/*.png')
# TODO play with these values to see how your classifier
# performs under different binning scenarios
orient=9
pix_per_cell=8
cell_per_block=2
hog_channel= 'ALL'
spatial = 32
histbin = 32
color_space = 'YUV'
samples = 300


car_features = extract_features(cars, color_space=color_space, spatial_size=(spatial, spatial),
                        hist_bins=histbin, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel)
notcar_features = extract_features(notcars, color_space=color_space, spatial_size=(spatial, spatial),
                        hist_bins=histbin, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel)

# Create an array stack of feature vectors
print('car_features', len(car_features))
print('notcar_features', len(notcar_features))
X = np.vstack((car_features, notcar_features)).astype(np.float64)
print(len(X))
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using spatial binning of:',spatial,
    'and', histbin,'histogram bins')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
dist = {'svc': svc,
            'X_scaler': X_scaler,
            'orient': orient,
            'pix_per_cell': pix_per_cell,
            'cell_per_block': cell_per_block,
            'spatial_size': (spatial, spatial),
            'hist_bins': histbin,
            'Training Time': round(t2 - t, 2),
            'color_space': color_space}
pickle.dump(dist, open("model_save/dist3.p", "wb"))
pred = svc.predict(X_test)
print('precision:{0:.3f}'.format(precision_score(y_test, pred)))
print('racall:{0:.3f}'.format(recall_score(y_test, pred)))
print('fscore:{0:.3f}'.format(f1_score(y_test, pred)))
print()

"""
from sklearn import svm, grid_search
# Use a linear SVC
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = grid_search.GridSearchCV(svc, parameters)

# Check the training time for the SVC
t=time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

dist = {'svc': svc,
        'clf': clf,
            'X_scaler': X_scaler,
            'orient': orient,
            'pix_per_cell': pix_per_cell,
            'cell_per_block': cell_per_block,
            'spatial_size': (spatial, spatial),
            'hist_bins': histbin,
            'Training Time': round(t2 - t, 2),
            'color_space': color_space}
pickle.dump(dist, open("model_save/dist2.p", "wb"))


pred = clf.predict(X_test)

print('precision:{0:.3f}'.format(precision_score(y_test, pred)))
print('racall:{0:.3f}'.format(recall_score(y_test, pred)))
print('fscore:{0:.3f}'.format(f1_score(y_test, pred)))
print()

"""
