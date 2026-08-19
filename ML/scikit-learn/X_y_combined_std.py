from iris_sickit_learn import X_train_std, X_test_std, y_train, y_test
import numpy as np

X_combined_std = np.vstack((X_train_std, X_test_std)) #Объединяет массивы по вертикали
y_combined = np.hstack((y_train, y_test)) #Объединяет массивы по горизонтали