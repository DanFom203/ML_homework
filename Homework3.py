import pygame as pygame
from math import hypot


def dbscan_naive(P, eps, minPts, distance):
    RED, YELLOW = -1, 0
    visited_points = set()
    clusters = {RED: [], YELLOW: []}

    def region_query(p):
        return [q for q in P if distance(p, q) < eps]

    def expand_cluster(p, neighbours, cluster_id):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(p)
        while neighbours:
            q = neighbours.pop()
            if q not in visited_points:
                visited_points.add(q)
                q_neighbours = region_query(q)
                if len(q_neighbours) >= minPts:
                    neighbours.extend(q_neighbours)
            if q not in clusters:
                clusters[cluster_id].append(q)

    for p in P:
        if p in visited_points:
            continue
        visited_points.add(p)
        neighbours = region_query(p)
        if len(neighbours) == 1:
            clusters[RED].append(p)
            continue
        elif len(neighbours) < minPts:
            clusters[YELLOW].append(p)
            continue
        else:
            clusters[p] = []
            expand_cluster(p, neighbours, p)
    return clusters


def main():
    points = []
    r = 10
    minPts, eps = 4, 4 * r
    colors = ['red', 'yellow', 'blue', 'green', 'purple', 'grey', 'pink', 'orange']
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    screen.fill('white')
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    point = pygame.mouse.get_pos()
                    points.append(point)
                    pygame.draw.circle(screen, 'black', point, r)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    labels = dbscan_naive(points, eps, minPts, lambda x, y: hypot(x[0] - y[0], x[1] - y[1]))
                    print(labels)
                    for i, group in enumerate(labels.values()):
                        for point in group:
                            pygame.draw.circle(screen, colors[i], point, r)

        pygame.display.flip()
    pygame.quit()


if __name__ == '__main__':
    main()
