#!/usr/bin/env python3
"""
Kaden Culbertson (krc6m9)
12/3/2025
Lab 6 Part 1.2

Route Planner with Dijkstra, modified for priority

Note: the edges were reversed to make it an undirected graph, as it I assumed it does not make sense for a route plotter to not be able to go in reverse, but
if you want it directed, simply delete in edges.csv where the edges flip (2,1,2.5 and below), but beware this can cause infinite distance

General Usage: python3 route_planner_task1_2.py ../data/nodes.csv ../data/edges.csv ../data/priority.csv 1 <max_difference>
where max_difference is the percent more a higher priority dstance can be over a lower priority distance

Example Usage: 
Keep With High Priority: python3 route_planner_task1_2.py ../data/nodes.csv ../data/edges.csv ../data/priority.csv 1 1.2
Keep With Some High Priority: python3 route_planner_task1_2.py ../data/nodes.csv ../data/edges.csv ../data/priority.csv 1 0.6
Switch To Lower Priority: python3 route_planner_task1_2.py ../data/nodes.csv ../data/edges.csv ../data/priority.csv 1 0.2
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

# a priority node that contains the id and the string value
class PriorityNode:
    def __init__(self, node_id: int, priority: str):
        self.id = node_id
        self.priority_value = priority
        self.done = False

# holds all of the priorities
class Priority:
    def __init__(self):
        self.priorities: Dict[int, PriorityNode] = {}
    
    def add_priority_node(self, node_id: int, priority: str):
        self.priorities[node_id] = PriorityNode(node_id, priority)


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
            priority_str = row['priority']
            priority.add_priority_node(node_id, priority_str)
    
    return priority


def main():
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <nodes.csv> <edges.csv> <priority.csv> <start_node> <max_difference (percent more a higher priority dstance can be over a lower priority distance)>")
        sys.exit(1)
    
    nodes_file = sys.argv[1]
    edges_file = sys.argv[2]
    priority_file = sys.argv[3]
    start_node = int(sys.argv[4])

    
    # Load graph
    graph = load_graph(nodes_file, edges_file)

    priority = load_priority(priority_file)

    current_node = start_node

    old_current_node = start_node

    # percent more a higher priority dstance can be over a lower priority distance
    max_difference = 1 + float(sys.argv[5])

    # counters for the total nodes and distances
    total_nodes = 0

    total_distance = 0
    
    # Validate nodes
    if start_node not in graph.nodes:
        print("Invalid start node")
        sys.exit(1)
    
    while True:

        # used to check constrainsts, initialied to -1 in case of priority not existing
        lowest_high_dist = -1
        lowest_med_dist = -1
        lowest_low_dist = -1
        lowest_high_id = -1
        lowest_med_id = -1
        lowest_low_id = -1

        for priority_node in priority.priorities.values():
            
            # skip if you already went there
            if priority_node.done:
                continue

            dist, prev, nodes_explored = dijkstra(graph, current_node, priority_node.id)
            
            if priority_node.priority_value == 'HIGH':
                if lowest_high_dist == -1 or lowest_high_dist > dist[priority_node.id]:
                    lowest_high_dist = dist[priority_node.id]
                    lowest_high_id = priority_node.id
            elif priority_node.priority_value == 'MEDIUM':
                if lowest_med_dist == -1 or lowest_med_dist > dist[priority_node.id]:
                    lowest_med_dist = dist[priority_node.id]
                    lowest_med_id = priority_node.id
            elif priority_node.priority_value == 'LOW':
                if lowest_low_dist == -1 or lowest_low_dist > dist[priority_node.id]:
                    lowest_low_dist = dist[priority_node.id]
                    lowest_low_id = priority_node.id
            else:
                print(f"Invalid priority value of {priority_node.priority_value}")
                sys.exit(1)

        if (lowest_med_dist == -1 or lowest_high_dist < lowest_med_dist * max_difference) and (lowest_low_dist == -1 or lowest_high_dist < lowest_low_dist * max_difference) and lowest_high_dist != -1:
            dist, prev, nodes_explored = dijkstra(graph, current_node, lowest_high_id)
            priority.priorities[lowest_high_id].done = True
            current_node = lowest_high_id
        elif (lowest_low_dist == -1 or lowest_med_dist < lowest_low_dist * max_difference) and lowest_med_dist != -1:
            dist, prev, nodes_explored = dijkstra(graph, current_node, lowest_med_id)
            priority.priorities[lowest_med_id].done = True
            current_node = lowest_med_id
        elif lowest_low_dist != -1:
            dist, prev, nodes_explored = dijkstra(graph, current_node, lowest_low_id)
            priority.priorities[lowest_low_id].done = True
            current_node = lowest_low_id
        else:
            break # if there is nowwhere left to visit
                

        '''
        Idea: find the dijsktas to all, keep the lowest id and distance for each of the three categories (6 vars total)
        do the check if the closest high meets criteria, if not closest medium, then closest small (eg check if medium or small has smaller distace by some percentage, then if small has smaller by some percentage than medium)
        run dikstras one more time to that node, also checking off the visited bool in the priority node list (its dictionary)
        print this path and the other stuff
        repeat as long as there are still locations to visit
        '''

        total_nodes += nodes_explored
        total_distance += dist[current_node]

        # Print results
        print(f"Priority: {priority.priorities[current_node].priority_value}")
        print_path(graph, prev, old_current_node, current_node, dist[current_node])
        print(f"Nodes explored: {nodes_explored}. Current total: {total_nodes}")
        old_current_node = current_node
    print(f"Total nodes explored: {total_nodes} with a total distance of: {total_distance}")


if __name__ == "__main__":
    main()