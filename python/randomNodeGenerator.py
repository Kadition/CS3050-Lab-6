from math import floor
import random
import csv

def generate_nodes(n, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'lat', 'lon', 'earliest', 'latest'])
        for i in range(1, n + 1):
            lat = round(random.uniform(38.9000, 38.9200), 4)
            lon = round(random.uniform(-77.0450, -77.0250), 4)
            earliest = random.randint(0, 100)
            latest = earliest + random.randint(10, 50)
            writer.writerow([i, lat, lon, earliest, latest])

def generate_edges(n_nodes, n_edges, filename):
    edges = set()
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['from', 'to', 'distance'])
        while len(edges) < n_edges:
            a, b = random.sample(range(1, n_nodes + 1), 2)
            edge = (a, b)
            if edge not in edges and (b, a) not in edges:
                distance = round(random.uniform(5.0, 100.0), 1)
                writer.writerow([a, b, distance])
                edges.add(edge)

if __name__ == "__main__":
    n_nodes = 10000
    n_edges = floor(n_nodes * 1.30)
    generate_nodes(n_nodes, f"random_nodes_{n_nodes}.csv")
    generate_edges(n_nodes, n_edges, f"random_edges_{n_edges}.csv")
    print("Files 'random_nodes.csv' and 'random_edges.csv' generated.")