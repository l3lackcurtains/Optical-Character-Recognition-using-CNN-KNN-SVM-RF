import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import helpers
import image_detection as detector
from collections import Counter
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

print('Start loading data.')
files, labels = helpers.load_chars74k_data()
X, y = helpers.create_dataset(files, labels)
print('Data has been loaded.')
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2, train_size=0.82)

# Normalizing images.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('\nKNN Classifier with n_neighbors = 5, algorithm = auto, n_jobs = 10')
print('\n***** Recording time *****')
e1 = cv2.getTickCount()
clf = KNeighborsClassifier(n_neighbors=5,algorithm='ball_tree',n_jobs=10)
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

e2 = cv2.getTickCount()
time0 = (e2 - e1) / cv2.getTickFrequency()
print('\n ***** Total time elapsed:',time0, ' *****')

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
pl.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
pl.xlabel('Predicted')
pl.ylabel('True')
pl.show()

detection2 = './detection-images/detection-5.jpg'
samples2 = detector.sliding_window(detection2)
samples_tf2 = samples2.astype('float32')
print('\nStart detection on example image: ', detection2)
predictions2 = clf.predict(samples_tf2)
value_list2 = []

for pred2 in predictions2:
	value_list2.append(helpers.num_to_char(pred2))

predictCount = Counter(value_list2)
print('\nPrediction result', predictCount)

print('\nResults in Probability\n')
for k, v in predictCount.items():
	print(k, ':', v/len(predictions2))

out = max(value_list2,key=value_list2.count)
print('\nMost Predicted Character is', out)

resImg = ''
if out == 'e':
	resImg = './detection-images/detection-5.jpg'
if out == 'a':
	resImg = './detection-images/detection-4.jpg'

if resImg:
	img = cv2.imread(resImg,0)
	cv2.imshow("Detected Image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
