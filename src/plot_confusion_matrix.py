mport matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.transforms import Bbox
import numpy as np

plt.rcParams.update({'font.size': 50})

def f(num):
    return round(num/total,2)
tp = 60075
tn = 368890
fn = 4947
fp = 8959
total = 442871
cm = np.array([[tn,fp],[fn,tp]])
labels = np.array(["unmodified","modified"])
sensitivity = tp / (tp+fn)
specificity = tn / (tn+fp)
accuracy = (tp + tn)/(tp+tn+fp+fn)
negativePV = tn / (tn+fn)
precision = tp / (tp+fp)

print("sensitivity: ",sensitivity)
print("specificity: ",specificity)
print("accuracy: ",accuracy)
print("negativePV: ",negativePV)
print("precision: ",precision)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels = labels)
fig, ax = plt.subplots(figsize=(30,15))
disp.plot(values_format='',ax=ax)

#bbox_config = Bbox([[-30, -30], [30, 30]])
#for labels in disp.text_:
#    labels.set_fontsize(20)
#plt.ticklabel_format(style='plain')    # to prevent scientific notation.
plt.savefig('confusion_matrix_img.png', transparent=True)
plt.show()
