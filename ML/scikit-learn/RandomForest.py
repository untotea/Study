from sklearn.ensemble import RandomForestClassifier
from iris_sickit_learn import X_train, y_train
from decision_tree import X_combined, y_combined
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt
forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.xlabel('длина лепестка [см]')
plt.ylabel('ширина лепестка [см]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()