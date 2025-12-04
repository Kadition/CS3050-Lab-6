#!/usr/bin/env python3
"""
Kaden Culbertson (krc6m9)
12/4/2025
Lab 6 Part 1.2

Route Planner with Dijkstra, modified for priority
"""

import sys
import csv
import heapq
import math
from typing import Dict, List, Tuple, Optional

EARTH_RADIUS = 6371.0  # km


class Node:
    """Represents a node in the graph"""
    def __init__(self, node_id: int, lat: float, lon: float):
        self.id = node_id
        self.lat = lat
        self.lon = lon


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
    
    def add_node(self, node_id: int, lat: float, lon: float):
        """Add a node to the graph"""
        self.nodes[node_id] = Node(node_id, lat, lon)
        if node_id not in self.adj_list:
            self.adj_list[node_id] = []
    
    def add_edge(self, from_id: int, to_id: int, weight: float):
        """Add an edge to the graph"""
        if from_id not in self.adj_list:
            self.adj_list[from_id] = []
        self.adj_list[from_id].append(Edge(to_id, weight))

class PriorityNode:
    def __init__(self, node_id: int, priority: str):
        self.id = node_id
        self.priority_value = priority
        self.done = False

class Priority:
    def __init__(self):
        self.priorities: Dict[int, PriorityNode] = {}
    
    def add_priority_node(self, node_id: int, priority: str):
        self.nodes[node_id] = Node(node_id, priority)


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
            
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
    
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
    print(f"Distance in this path: {distance:.2f} km")


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
            graph.add_node(node_id, lat, lon)
    
    # Load edges
    with open(edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_id = int(row['from'])
            to_id = int(row['to'])
            distance = float(row['distance'])
            graph.add_edge(from_id, to_id, distance)
    
    return graph

def load_priority(priority_file: str) -> Priority:
    """Load graph from CSV files"""
    priority = Priority()
    
    # Load priority
    with open(priority_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row['id'])
            priority_str = int(row['priority'])
            priority.add_priority_node(node_id, priority_str)
    
    return priority


def main():
    if len(sys.argv) != 7:
        print(f"Usage: {sys.argv[0]} <nodes.csv> <edges.csv> <start_node> <end_node> <algorithm> <priority.csv")
        print("Algorithms: dijkstra, astar, bellman-ford")
        sys.exit(1)
    
    nodes_file = sys.argv[1]
    edges_file = sys.argv[2]
    start_node = int(sys.argv[3])
    end_node = int(sys.argv[4])
    algorithm = sys.argv[5]
    priority_file = sys.argv[6]
    
    # Load graph
    graph = load_graph(nodes_file, edges_file)

    priority = load_priority(priority_file)
    
    # Validate nodes
    if start_node not in graph.nodes or end_node not in graph.nodes:
        print("Invalid start or end node")
        sys.exit(1)
    
    # Run selected algorithm
    if algorithm == "dijkstra":
        for i in range(0, len(priority_file)):

            found = False

            for priority_node in priority.priorities:
                if priority_node.priority_value == "HIGH" and priority_node.done == False:
                    found = True
                    priority_node.done = True
                    

            '''
            Idea: find the dijsktas to all, keep the lowest id and distance for each of the three categories (6 vars total)
            do the check if the closest high meets criteria, if not closest medium, then closest small (eg check if medium or small has smaller distace by some percentage, then if small has smaller by some percentage than medium)
            run dikstras one more time to that node, also checking off the visited bool in the priority node list (its dictionary)
            print this path and the other stuff
            repeat as long as there are still locations to visit
            '''

            total_nodes = 0
            total_distance = 0
            dist, prev, nodes_explored = dijkstra(graph, start_node, end_node)
            total_nodes += nodes_explored
            total_distance += dist
            # Print results
            print()
            print_path(graph, prev, start_node, end_node, dist[end_node])
            print(f"Nodes explored: {nodes_explored}. Current total: {total_nodes}")
        print(f"Total nodes explored: {total_nodes} with a total distance of: {total_distance}")
    else:
        print(f"Unknown algorithm: {algorithm}")
        print("Available algorithms: dijkstra, astar, bellman-ford")
        sys.exit(1)


if __name__ == "__main__":
    main()