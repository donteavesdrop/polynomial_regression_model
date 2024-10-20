import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Создание массива с предоставленными данными
data = np.array([
    [0.904, 75.5, 25.2, 3343, 77],
    [0.922, 78.5, 21.8, 3001, 78.2],
    [0.763, 78.4, 25.7, 3101, 68],
    [0.923, 77.7, 17.8, 3543, 77.2],
    [0.918, 84.4, 15.9, 3237, 77.2],
    [0.906, 75.9, 22.4, 3330, 77.2],
    [0.905, 76.0, 20.6, 3808, 75.7],
    [0.545, 67.5, 25.2, 2415, 62.6],
    [0.894, 78.2, 20.7, 3295, 78],
    [0.9, 78.1, 17.5, 3504, 78.2],
    [0.932, 78.6, 19.7, 30565, 79],
    [0.74, 84.0, 18.5, 3007, 67.6],
    [0.701, 59.2, 54.4, 2844, 69.8],
    [0.744, 90.2, 23.0, 2861, 68.4],
    [0.921, 72.8, 20.2, 3259, 77.9],
    [0.927, 67.7, 25.2, 3350, 78.1],
    [0.802, 82.6, 22.4, 3344, 72.5],
    [0.747, 74.4, 22.7, 2704, 66.6],
    [0.927, 83.3, 18.1, 3642, 76.7],
    [0.721, 83.7, 20.1, 2753, 68.8],
    [0.913, 73.8, 17.3, 2916, 76.8],
    [0.918, 79.2, 16.8, 3551, 78.1],
    [0.833, 71.5, 29.9, 3177, 73.9],
    [0.914, 75.3, 20.3, 3280, 78.6],
    [0.923, 79.0, 14.1, 3160, 78.5]
])

# Разделение данных на признаки (X) и целевую переменную (y)
X = data[:, 1:]
y = data[:, 0]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5, shuffle=False)

# Обучающая выборка
X_train = X[:20]
y_train = y[:20]

# Тестовая выборка
X_test = X[20:]
y_test = y[20:]

# Обучение модели многорядного полиномиального алгоритма
degree = 1  # степень полинома
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X_train, y_train)

# Предсказание значений для тестовой выборки
y_test_pred = model.predict(X_test)

# Оценка модели на тестовой выборке
test_mse = mean_squared_error(y_test, y_test_pred)

# Вывод результатов
print("Среднеквадратическая ошибка на тестовой выборке:", test_mse)
# Вывод предсказанных и истинных значений на тестовой выборке
print("Предсказанные значения на тестовой выборке:", y_test_pred)
print("Истинные значения на тестовой выборке:", y_test)

# Рассчет средней ошибки аппроксимации
approximation_error = np.abs(y_test - y_test_pred).mean()
print("Средняя ошибка аппроксимации на тестовой выборке:", approximation_error)

# Вывод качества обученной модели
if approximation_error < 0.05:
    print("Модель хорошо аппроксимирует данные.")
else:
    print("Модель плохо аппроксимирует данные.")

# Построение графика значений исходной модели и модели, построенной по МГУА
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Исходная модель')
plt.plot(y_test_pred, label='Модель МГУА')
plt.xlabel('Номер образца')
plt.ylabel('Значение модели')
plt.title('Сравнение исходной модели и модели, построенной по МГУА')
plt.legend()
plt.grid(True)
plt.show()
