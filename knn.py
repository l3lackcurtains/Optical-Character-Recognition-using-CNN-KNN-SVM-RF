from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import style
from sklearn.model_selection import train_test_split

import helpers
import image_detection as detector

print('Start loading data.')
files, labels = helpers.load_chars74k_data()
X, y = helpers.create_dataset(files, labels)
print('Data has been loaded.')

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2, train_size=0.8)

# Normalizing images.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('\nKNN Classifier with n_neighbors = 5, algorithm = auto, n_jobs = 10')

clf = KNeighborsClassifier(n_neighbors=5,algorithm='auto',n_jobs=10)
clf.fit(x_train,y_train)

print('\nCalculating Accuracy of trained Classifier...')
acc = clf.score(x_test,y_test)

print('\nMaking Predictions on Validation Data...')
y_pred = clf.predict(x_test)

print('\nCalculating Accuracy of Predictions...')
accuracy = accuracy_score(y_test, y_pred)


print('\nClassifier Accuracy: ',acc)
print('\nPredicted Values: ',y_pred)
print('\nAccuracy: ',accuracy)


detection2 = './detection-images/detection-1.jpg'
samples2 = detector.sliding_window(detection2)
samples_tf2 = samples2.astype('float32')
print('Start detection on example image: ', detection2)
predictions2 = clf.predict(samples_tf2)
value_list2 = []

for pred2 in predictions2:
	value_list2.append(helpers.num_to_char(pred2))
    

print('Predicted values on', detection2, Counter(value_list2))