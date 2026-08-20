import matplotlib.pyplot as plt
import numpy as np
from X_y_combined_std import X_combined_std, y_combined, X_train_std, y_train
from plot_decision_regions import plot_decision_regions
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel('Длинна лепестка [стандатизированная]')
plt.ylabel('Ширина лепестка [стандатизированная]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()