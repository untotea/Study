from sklearn.svm import SVC
import matplotlib.pyplot as plt
from plot_decision_regions import plot_decision_regions
import numpy as np
from XOR_graphic import X_xor, y_xor

svm = SVC(kernel='rbf', random_state=1,  gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()