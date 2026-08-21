from sklearn.svm import SVC
from iris_sickit_learn import X_train_std, y_train
from plot_decision_regions import plot_decision_regions
import numpy as np
from X_y_combined_std import X_combined_std, y_combined
import matplotlib.pyplot as plt

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))

plt.xlabel('длина лепестка [стандартизированная]')
plt.ylabel('ширина лепестка [стандартизированная]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
