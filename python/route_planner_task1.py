#!/usr/bin/env python3
"""
Kaden Culbertson (krc6m9)
12/4/2025
Lab 6 Part 1

Route Planner with Dijkstra, modified for time frames

for a feasible path: python3 route_planner_task1.py ../data/example_time_window_nodes.csv ../data/task1_edges_success.csv 1 3 dijkstra
for an infeasible path: python3 route_planner_task1.py ../data/example_time_window_nodes.csv ../data/task1_edges_fail.csv 1 3 dijkstra
for a case where the shortest distance path violates constraints: python3 route_planner_task1.py ../data/example_time_window_nodes.csv ../data/task1_edges_violation.csv 1 3 dijkstra
"""

import sys
import csv
import heapq
import math
from typing import Dict, List, Tuple, Optional

EARTH_RADIUS = 6371.0  # km


class Node:
    """Represents a node in the graph"""
    def __init__(self, node_id: int, lat: float, lon: float, earliest: int, latest: int):
        self.id = node_id
        self.lat = lat
        self.lon = lon
        self.earliest = earliest # added new data to the nodes (earliest, latest, and distance)
        self.latest = latest


class Edge:
    """Represents an edge in the graph"""
    def __init__(self, to: int, weight: float):
        self.to = to
        self.weight = weight


class Graph:
    """Graph data structure with adjacency list"""
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.adj_list: Dict[int, List[Edge]] = {}
    
    def add_node(self, node_id: int, lat: float, lon: float, earliest: int, latest: int):
        """Add a node to the graph"""
        self.nodes[node_id] = Node(node_id, lat, lon, earliest, latest)
        if node_id not in self.adj_list:
            self.adj_list[node_id] = []
    
    def add_edge(self, from_id: int, to_id: int, weight: float):
        """Add an edge to the graph"""
        if from_id not in self.adj_list:
            self.adj_list[from_id] = []
        self.adj_list[from_id].append(Edge(to_id, weight))


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points"""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS * c


def dijkstra(graph: Graph, start: int, end: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]], int]:
    """
    Dijkstra's algorithm for shortest path
    Returns: (distances, previous nodes, nodes explored)
    """

    # gives the total amount a bypasses this iteration searching for a path
    ignore_count = 0

    # gives the total amount a bypasses it can use in the case of a fail
    ignore_count_amount = 0

    # if the path has failed
    failed = False

    # continue util you find a path or you are able to violate the constrait on every node, but there is still no path
    while ignore_count_amount < len(graph.nodes):
        
        ignore_count = ignore_count_amount

        dist = {node_id: float('inf') for node_id in graph.nodes}
        prev = {node_id: None for node_id in graph.nodes}
        dist[start] = 0
        
        pq = [(0, start)]
        nodes_explored = 0
        visited = set()
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            
            visited.add(u)
            nodes_explored += 1
            
            if u == end:
                break
            
            if current_dist > dist[u]:
                continue
            
            for edge in graph.adj_list.get(u, []):
                v = edge.to
                alt = dist[u] + edge.weight
                
                # the second two checks will make sure that the time meets the constraints that the earliest and latest attributes provide
                # or, if it has failed, it can bypass that to a certain amount, guarenting the least amount of violations
                if alt < dist[v] and ((graph.nodes[v].earliest <= alt and graph.nodes[v].latest >= alt) or ignore_count > 0):
                    if(not (graph.nodes[v].earliest <= alt and graph.nodes[v].latest >= alt)):
                        ignore_count -= 1
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, v))

        # a check to see if a path was not found
        if prev[end] is None and start != end and not failed:
            print("No feasible path satisfying time constraints")
            print("Failed on node " + str(u))
            print("Finding path with least number of violations")
            ignore_count_amount += 1
            failed = True
        
        # checks if the path failed again
        elif prev[end] is None and start != end and failed:
            ignore_count_amount += 1

        # if you have a path, return it
        else:
            break
        
    return dist, prev, nodes_explored

def reconstruct_path(prev: Dict[int, Optional[int]], start: int, end: int) -> Optional[List[int]]:
    """Reconstruct path from start to end using previous nodes"""
    if prev[end] is None and start != end:
        return None
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev[current]
    
    path.reverse()
    return path


def print_path(graph: Graph, prev: Dict[int, Optional[int]], start: int, end: int, distance: float):
    """Print the path from start to end"""
    path = reconstruct_path(prev, start, end)
    
    if path is None:
        print("No path found")
        return
    
    path_str = " -> ".join(str(node) for node in path)
    print(f"Path from {start} to {end}: {path_str}")
    print(f"Total distance: {distance:.2f} km")


def load_graph(nodes_file: str, edges_file: str) -> Graph:
    """Load graph from CSV files"""
    graph = Graph()
    
    # Load nodes
    with open(nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row['id'])
            lat = float(row['lat'])
            lon = float(row['lon'])
            earliest = float(row['earliest']) # added new data to the nodes (earliest, latest, and distance)
            latest = float(row['latest'])
            graph.add_node(node_id, lat, lon, earliest, latest)
    
    # Load edges
    with open(edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_id = int(row['from'])
            to_id = int(row['to'])
            distance = float(row['distance'])
            graph.add_edge(from_id, to_id, distance)
    
    return graph


def main():
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <nodes.csv> <edges.csv> <start_node> <end_node> <algorithm>")
        print("Algorithms: dijkstra, astar, bellman-ford")
        sys.exit(1)
    
    nodes_file = sys.argv[1]
    edges_file = sys.argv[2]
    start_node = int(sys.argv[3])
    end_node = int(sys.argv[4])
    algorithm = sys.argv[5]
    
    # Load graph
    graph = load_graph(nodes_file, edges_file)
    
    # Validate nodes
    if start_node not in graph.nodes or end_node not in graph.nodes:
        print("Invalid start or end node")
        sys.exit(1)
    
    # Run selected algorithm
    if algorithm == "dijkstra":
        print("=== Dijkstra's Algorithm ===")
        dist, prev, nodes_explored = dijkstra(graph, start_node, end_node)
    else:
        print(f"Unknown algorithm: {algorithm}")
        print("Available algorithms: dijkstra, astar, bellman-ford")
        sys.exit(1)
    
    # Print results
    print_path(graph, prev, start_node, end_node, dist[end_node])
    print(f"Nodes explored: {nodes_explored}")


if __name__ == "__main__":
    main()