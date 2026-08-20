import matplotlib.pylab as plt
import numpy as np

np.random.seed(1) #Генерирует всегда одинаковые случайные числа не смотря на разные запуски.
X_xor = np.random.randn(200, 2) #создает двумерный массив размером 200 строк и 2 столбца. Он заполняет этот массив случайными числами из стандартного нормального распределения (среднее значение равно 0, а разброс/дисперсия равна 1).
y_xor = np.logical_xor(X_xor[:, 0] > 0, #Возращает True если лишь одно значение истино.
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1) #Проверяет значение y_xor
plt.scatter(X_xor[y_xor==1, 0],
            X_xor[y_xor==1, 1], 
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.tight_layout()
plt.show()
