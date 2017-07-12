import numpy as np
from matplotlib import pyplot as plt
from  sklearn import metrics


def plot_roc_curve(fpr, tpr):
    plt.figure(1)
    plt.plot(fpr, tpr, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curve ROC")



y_true = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0]
y_true2 = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]

y_score = np.linspace(1,0, len(y_true))

[fpr, tpr, thresholds_roc] = metrics.roc_curve(y_true, y_score)
[fpr2, tpr2, thresholds_roc2] = metrics.roc_curve(y_true2, y_score)

plot_roc_curve(fpr, tpr)
plot_roc_curve(fpr2, tpr2)
plt.show()

print(fpr.shape)
print(fpr2.shape)
print(y_true)
print(y_score)

out = [3,4,5,5,6]
out2 = ",".join([str(value) for value in out.tolist()])
out3 = [float(value) for value in out2.split(",")]


