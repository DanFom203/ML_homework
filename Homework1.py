import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Загрузка данных ирисов
iris = load_iris()
X = iris.data

# Список для сохранения значений инерции
inertia_values = []

# Попробуем количество кластеров от 2 до 10
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Визуализация метода локтя
plt.figure(figsize=(7, 5))
plt.plot(range(2, 11), inertia_values, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Найдем оптимальное количество кластеров (номер, при котором происходит снижение скорости убывания инерции)
diffs = np.diff(inertia_values)  # Вычисляем разницу между соседними значениями инерции

# Задаем порог для снижения скорости убывания инерции
threshold = 0.1  # Произвольно выбранный порог

# Ищем момент, когда скорость убывания инерции становится меньше порога
optimal_k_index = np.where(diffs < threshold)[0][0] + 2  # Добавляем 2, чтобы сдвинуться на 2 индекса вперед, так как у нас есть разница на один элемент меньше

optimal_k = optimal_k_index + 1  # Добавляем 1, чтобы получить фактическое количество кластеров

print("Optimal number of clusters:", optimal_k)