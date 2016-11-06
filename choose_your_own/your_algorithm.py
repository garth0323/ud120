#!/usr/bin/python
import logging
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

# features_train = features_train[:len(features_train)/2] 
# labels_train = labels_train[:len(labels_train)/2] 
# features_test = features_test[:len(features_test)/2] 
# labels_test = labels_test[:len(labels_test)/2]


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]
logging.warning('initial visualization')
#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
logging.warning('end initial visualization')
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

logging.warning('here?')
from sklearn.ensemble import AdaBoostClassifier
logging.warning('imported')
clf = AdaBoostClassifier(n_estimators=50)
logging.warning('initialized')
clf = clf.fit(features_train, labels_train)
logging.warning('training')
pred = clf.predict(features_test)
logging.warning('predictions')

from sklearn.metrics import accuracy_score
logging.warning('imported accuracy')
accuracy = accuracy_score(pred, labels_test)
logging.warning('tested accuracy')
print "accuracy:", accuracy

try:
	logging.warning('pretty picture?')
  	prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
