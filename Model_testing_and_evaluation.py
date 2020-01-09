# Loading our saved model 
from tensorflow.keras.models import load_model
model = load_model('MCDNN.h5')

model.summary()

# checking the model metrics
model.metrics_names
# Evaluating the loading model
model.evaluate(x=x_test,y=y_test)

### Part 4: Model Testing

y_score = model.predict(x_test)
# Displaying the valuation metrics i.e ROC curve
import matplotlib.pyplot as plt
%matplotlib inline
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('MCDNN accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MCDNN loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss.png')
plt.show()

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2
n_classes = 23
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(8,8))

plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)


plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','ivory','MintCream','DarkGrey','Green', 'SlateBlue','PaleTurquoise','Orchid','SkyBlue','DarkOliveGreen','DarkGoldenRod','DarkTurquoise','DodgerBlue','Gold','DarkCyan','HotPink'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('All ROC curves of 23 classes ')
plt.legend(loc="lower right")
plt.savefig('all_roc.png')

plt.show()


# Zoom in view of the upper left corner.
plt.figure(2,figsize=(8,8))
plt.xlim(0, 0.005)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','ivory','MintCream','DarkGrey','Green', 'SlateBlue','PaleTurquoise','Orchid','SkyBlue','DarkOliveGreen','DarkGoldenRod','DarkTurquoise','DodgerBlue','Gold','DarkCyan','HotPink'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Zoom in view of the upper left corner of ROC curve')
plt.legend(loc="lower right")
plt.savefig('zoomed_roc.png')
#plt.show()

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

y_pred = model.predict(x_test)
y_pred.shape
pred = np.argmax(y_pred,axis=1) # raw probabilities to chosen class (highest probability)
pred.shape
y_compare = np.argmax(y_test,axis=1)
y_compare.shape
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score)) 

from IPython.display import display

# Don't display numpy in scientific notation
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# Generate predictions
pred = model.predict(x_test)

print("Numpy array of predictions")
display(pred[0:5])

print("As percent probability")
print(pred[0]*100)

score = metrics.log_loss(y_test, pred)
print("Log loss score: {}".format(score))

pred = np.argmax(y_pred,axis=1) # raw probabilities to chosen class (highest probability)

# Classification report

from sklearn.metrics import classification_report

predictions_test = model.predict_classes(x_test)
predictions_train = model.predict_classes(x_train)
y_test=np.argmax(y_test, axis=1)
y_train=np.argmax(y_train, axis=1)
data = df.groupby('label')['label'].count()
Training_Pred = (classification_report(y_train,predictions_train))
Testing_Pred = (classification_report(y_test,predictions_test))
print(Training_Pred)
print(Testing_Pred)

