from sklearn.neighbors import KNeighborsClassifier
from iris_sickit_learn import X_train_std, y_train
import matplotlib.pyplot as plt
from plot_decision_regions import plot_decision_regions
from X_y_combined_std import X_combined_std, y_combined

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))
plt.xlabel('длина лепестка [стандартизированная]')
plt.ylabel('ширина лепестка [стандартизированная]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()