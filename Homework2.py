import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def k_means(X, n_clusters, max_iter=100):
    # Инициализация центроидов случайными точками


    #centroids = X[np.random.choice(X.shape[0], size=n_clusters, replace=False)]
    centroids = [[4., 3.],[5., 2.],[6., 2.]]
    #centroids = ([-10, 10], [10, 10], [10, -10]]
    # Для визуализации
    colors = ['r', 'g', 'b', 'k']
    plt.figure(figsize=(10, 6))

    for i in range(max_iter):
        # Вычисление расстояний от каждой точки до центроидов и определение кластера
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Визуализация текущего состояния кластеров
        plt.figure(figsize=(10, 6))
        for cluster in range(n_clusters):
            plt.scatter(X[labels == cluster][:, 0], X[labels == cluster][:, 1], c=colors[cluster],
                        label=f'Cluster {cluster}')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='k', label='Centroids')
        plt.title(f"Iteration {i + 1}")
        plt.legend()
        plt.show()

        # Обновление центроидов
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])

        # Проверка сходимости
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


if __name__ == '__main__':
    # Загрузка датасета ирисов
    iris = load_iris()
    X = iris.data

    # Задаем количество кластеров, итераций и запускаем алгоритм
    n_clusters = 3  # Оптимальное количество кластеров, найденное ранее
    max_iter = 6
    centroids, labels = k_means(X[:, :2], n_clusters)
