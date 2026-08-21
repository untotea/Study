from sklearn.tree import DecisionTreeClassifier
from iris_sickit_learn import X_train, y_train, X_test, y_test
import numpy as np
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

tree_model = DecisionTreeClassifier(criterion='gini',
                                    max_depth=4,
                                    random_state=1)

tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree_model, test_idx=range(105, 150))
plt.xlabel('длина лепестка [см]')
plt.ylabel('ширина лепестка [см]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

from sklearn import tree
tree.plot_tree(tree_model)
plt.show()