from collections import deque
import random
import time
import numpy as np
from typing import Optional, Any, List, Dict, Set, Tuple, Generator
import scipy as sp

class Graph:
    """
    Graph class
    """
    def __init__(self):
        self._graph = {}
        self._index = 0
        self._stack = []
        self._indices = {}
        self._low_links = {}
        self._on_stack = set()
        self._sccs = []

    def add_vertex(self, vertex: str, data: Optional[Any]=None) -> None:
        """
        Adds a vertex to the graph
        :param vertex: the vertex name
        :param data: data associated with the vertex
        """
        if vertex not in self._graph:
            self._graph[vertex] = {'data': data, 'neighbors': {}}

    def add_edge(self, vertex1: str, vertex2: str, data: Optional[Any]=None) -> None:
        """
        Adds an edge to the graph
        :param vertex1: vertex1 key
        :param vertex2: vertex2 key
        :param data: the data associated with the vertex
        """
        if not vertex1 in self._graph or not vertex2 in self._graph:
            raise ValueError("The vertices do not exist")
        self._graph[vertex1]['neighbors'][vertex2] = data

    def get_neighbors(self, vertex: str) -> List[str]:
        """
        Get the list of vertex neighbors
        :param vertex: the vertex to query
        :return: the list of neighbor vertices
        """
        if vertex in self._graph:
            return list(self._graph[vertex]['neighbors'].keys())
        else:
            return []

    def vertex_exists(self, vertex: str) -> bool:
        """
        If contains a vertex
        :param vertex: the vertex name
        :return: boolean
        """
        return vertex in self._graph

    def edge_exists(self, vertex1: str, vertex2: str) -> bool:
        """
        If contains an edge
        :param vertex1: the vertex1 name
        :param vertex2: the vertex2 name
        :return: boolean
        """
        return vertex1 in self._graph and vertex2 in self._graph[vertex1]['neighbors']

    def connected_components(self, undirected_graph: 'Graph') -> List[List[str]]:
        connected_components = []
        is_visited = {vertex: False for vertex in undirected_graph._graph}

        for vertex in undirected_graph._graph:
            if not is_visited[vertex]:
                component = self.find_connected_component(vertex, is_visited, undirected_graph)
                connected_components.append(component)

        return connected_components

    def find_connected_component(self, src: str, is_visited: Dict[str, bool], undirected_graph: 'Graph') -> List[str]:
        component = []
        stack = [src]

        while stack:
            vertex = stack.pop()
            if not is_visited[vertex]:
                is_visited[vertex] = True
                component.append(vertex)
                for neighbor in undirected_graph.get_neighbors(vertex):
                    if not is_visited[neighbor]:
                        stack.append(neighbor)

        return component
    
    def weakly_connected_components(self) -> List[List[str]]:
        undirected_graph = self.make_copy_graph_undirected()

        return self.connected_components(undirected_graph)
    
    def largest_weakly_connected_component(self) -> int:
        components = self.weakly_connected_components()
        return max(len(component) for component in components) if components else 0
    
    def number_of_weakly_connected_components(self) -> int:
        components = self.weakly_connected_components()
        return len(components)

    def bfs(self, start: str) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Perform BFS and return parent and distance dictionaries
        :param start: The start vertex
        :return: parent and distance dictionaries
        """
        par = {v: None for v in self._graph}
        dist = {v: float('inf') for v in self._graph}
        
        q = deque([start])
        dist[start] = 0
        
        while q:
            node = q.popleft()
            for neighbor in self.get_neighbors(node):
                if dist[neighbor] == float('inf'):
                    par[neighbor] = node
                    dist[neighbor] = dist[node] + 1
                    q.append(neighbor)
                    
        return par, dist

    # def print_shortest_path(self, start: str, end: str) -> None:
    #     """
    #     Print the shortest path from start to end
    #     :param start: The start vertex
    #     :param end: The end vertex
    #     """
    #     if not self.vertex_exists(start) or not self.vertex_exists(end):
    #         print("One or both vertices not found in the graph")
    #         return
        
    #     par, dist = self.bfs(start)
        
    #     if dist[end] == float('inf'):
    #         print("Source and Destination are not connected")
    #         return
        
    #     path = []
    #     current_node = end
    #     while current_node is not None:
    #         path.append(current_node)
    #         current_node = par[current_node]
        
    #     path.reverse()
    #     print(" -> ".join(path))

    def estimate_shortest_paths(self, samples: int) -> float:
        vertices = list(self._graph.keys())
        sampled_vertices = random.sample(vertices, samples)

        start_time = time.time()
        for vertex in sampled_vertices:
            self.bfs(vertex)
        end_time = time.time()

        return end_time - start_time
    
    def estimate_diameter(self, samples: int) -> float:
        """
        Estimate the diameter of the graph by sampling a subset of vertices and performing BFS
        from each sampled vertex to find the maximum distance to any other vertex.
        :param samples: The number of vertices to sample
        :return: An estimate of the diameter of the graph
        """
        vertices = list(self._graph.keys())

        max_distance = 0

        for _ in range(samples): # takes into account not landing on the same component
        
            vertex = random.choice(vertices)

            for _ in range(samples):
                _, dist = self.bfs(vertex)
                
                dist = {k: v for k, v in dist.items() if v < float('inf')}

                #get the maximum distance and the vertex that has it
                max_dist = max(dist.values())

                max_vertex = max(dist, key=dist.get)

                # print(f"Vertex: {vertex}, Max Distance: {max_dist}, Max Vertex: {max_vertex}")

                #if the maximum distance is greater than the current maximum distance, update it
                if max_dist > max_distance:
                    max_distance = max_dist
                
                vertex = max_vertex

        return max_distance


    def page_rank(self, damping_factor: float = 0.85, max_iterations: int = 100, tol: float = 1.0e-6) -> Dict[str, float]:
        """
        Computes PageRank for each vertex in the graph.
        :param graph: the graph
        :param damping_factor: the damping factor (default 0.85)
        :param max_iterations: maximum number of iterations (default 100)
        :param tol: tolerance for convergence (default 1.0e-6)
        :return: a dictionary of vertex PageRank values
        """
        vertices = list(self._graph.keys())
        num_vertices = len(vertices)
        if num_vertices == 0:
            return {}

        # Initialize PageRank values
        page_rank_values = {vertex: 1.0 / num_vertices for vertex in vertices}

        #create L dictionary
        L = {vertex: len(self.get_neighbors(vertex)) for vertex in vertices}

        #create dict of vertices that reference the current vertex
        # R = {vertex: [v for v in vertices if self.edge_exists(v, vertex)] for vertex in vertices}

        transposed_graph = self.transpose()

        R = {vertex: list(transposed_graph.get_neighbors(vertex)) for vertex in transposed_graph._graph}

        for iteration in range(max_iterations):
            new_page_rank_values = {}
            for vertex in vertices:
                rank_sum = 0.0
                for v in R[vertex]:
                    rank_sum += page_rank_values[v] / L[v]
                new_page_rank_values[vertex] = (1 - damping_factor) / num_vertices + damping_factor * rank_sum

            # Check for convergence
            diff = sum(abs(new_page_rank_values[vertex] - page_rank_values[vertex]) for vertex in vertices)
            if diff < tol:
                print(f"Converged after {iteration + 1} iterations")
                break

            page_rank_values = new_page_rank_values

        return page_rank_values

    
    def transpose(self) -> 'Graph':
        """
        Returns the transpose of the graph
        :return: A new Graph object that is the transpose of the current graph
        """
        transposed = Graph()
        for vertex in self._graph:
            transposed.add_vertex(vertex, self._graph[vertex]['data'])
        for vertex in self._graph:
            for neighbor in self._graph[vertex]['neighbors']:
                transposed.add_edge(neighbor, vertex, self._graph[vertex]['neighbors'][neighbor])
        return transposed

    def count_and_list_cycle_triangles(self) -> Tuple[int, List[Tuple[str, str, str]]]:
        """
        Counts and optionally lists the cycle triangles in the graph.
        :return: A tuple containing the number of cycle triangles and the list of triangles
        """
        GT = self.transpose()
        c = 0
        triangles = []

        for u in self._graph:
            for v in self.get_neighbors(u):
                if u < v:
                    S = set(GT.get_neighbors(u)).intersection(self.get_neighbors(v))
                    for w in S:
                        if u < w:
                            triangles.append((u, v, w))
                            c += 1
                            
        return c, triangles

    def make_copy_graph_undirected(self):
        """
        Makes a copy of the graph as an undirected graph
        """
        undirected_graph = Graph()
        for vertex in self._graph:
            undirected_graph.add_vertex(vertex, self._graph[vertex]['data'])
        for vertex in self._graph:
            for neighbor in self._graph[vertex]['neighbors']:
                undirected_graph.add_edge(vertex, neighbor)
                undirected_graph.add_edge(neighbor, vertex)
        return undirected_graph
    
    def count_triangles(self) -> int:
        """
        Counts the number of triangles in the graph
        """
        undirected_graph = self.make_copy_graph_undirected()
        count = 0
        for vertex in self._graph:
            neighbors = undirected_graph.get_neighbors(vertex)
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if undirected_graph.edge_exists(neighbors[i], neighbors[j]):
                        count += 1
        return count // 3       
    
    # def dfs_find_cycles(self, start_vertex: str) -> List[int]:
    #     """
    #     Perform DFS from the start vertex to find all cycles including this vertex.
    #     :param start_vertex: The start vertex for the DFS
    #     :return: List of cycle lengths found
    #     """
    #     stack = [(start_vertex, [start_vertex])]
    #     cycles = []

    #     while stack:
    #         (vertex, path) = stack.pop()
    #         for neighbor in self.get_neighbors(vertex):
    #             if neighbor == start_vertex and len(path) > 2:
    #                 cycles.append(len(path))
    #             elif neighbor not in path:
    #                 stack.append((neighbor, path + [neighbor]))
        
    #     return cycles

    # def estimate_circumference(self, samples: int = 100) -> int:
    #     """
    #     Estimate the circumference of the graph by sampling a subset of vertices and performing DFS
    #     to find cycles that include the sampled vertices.
    #     :param samples: The number of vertices to sample
    #     :return: An estimate of the circumference of the graph
    #     """
    #     vertices = list(self._graph.keys())
    #     sampled_vertices = random.sample(vertices, min(samples, len(vertices)))
    #     max_cycle_length = 0

    #     for vertex in sampled_vertices:
    #         cycles = self.dfs_find_cycles(vertex)
    #         if cycles:
    #             max_cycle_length = max(max_cycle_length, max(cycles))

    #     return max_cycle_length

    # def _strong_connect(self, vertex: str) -> None:
    #     """
    #     Non-recursive helper function for Tarjan's algorithm to find strongly connected components
    #     """
    #     stack = [(vertex, 0)]
    #     visited = set()
        
    #     while stack:
    #         v, index = stack[-1]
            
    #         if v not in visited:
    #             visited.add(v)
    #             self._indices[v] = self._index
    #             self._low_links[v] = self._index
    #             self._index += 1
    #             self._stack.append(v)
    #             self._on_stack.add(v)
            
    #         neighbors = self.get_neighbors(v)
            
    #         if index < len(neighbors):
    #             neighbor = neighbors[index]
    #             stack[-1] = (v, index + 1)
                
    #             if neighbor not in self._indices:
    #                 stack.append((neighbor, 0))
    #             elif neighbor in self._on_stack:
    #                 self._low_links[v] = min(self._low_links[v], self._indices[neighbor])
    #         else:
    #             if self._low_links[v] == self._indices[v]:
    #                 scc = set()
    #                 while True:
    #                     w = self._stack.pop()
    #                     self._on_stack.remove(w)
    #                     scc.add(w)
    #                     if w == v:
    #                         break
    #                 self._sccs.append(scc)
    #             stack.pop()
    #             if stack:
    #                 w, _ = stack[-1]
    #                 self._low_links[w] = min(self._low_links[w], self._low_links[v])

    # def find_strongly_connected_components(self) -> List[Set[str]]:
    #     """
    #     Finds and returns all strongly connected components
    #     """
    #     self._index = 0
    #     self._stack = []
    #     self._indices = {}
    #     self._low_links = {}
    #     self._on_stack = set()
    #     self._sccs = []

    #     for vertex in self._graph:
    #         if vertex not in self._indices:
    #             self._strong_connect(vertex)

    #     return self._sccs

    # def largest_strongly_connected_component(self) -> int:
    #     """
    #     Returns the size of the largest strongly connected component
    #     """
    #     sccs = self.find_strongly_connected_components()
    #     return max(len(scc) for scc in sccs) if sccs else 0

    # def number_of_strongly_connected_components(self) -> int:
    #     """
    #     Returns the number of strongly connected components
    #     """
    #     sccs = self.find_strongly_connected_components()
    #     return len(sccs)
    
    def floyd(self, f, x0) -> (int, int):
        """Floyd's cycle detection algorithm."""
        tortoise = f(x0) # f(x0) is the element/node next to x0.
        hare = f(f(x0))
        while tortoise != hare:
            tortoise = f(tortoise)
            hare = f(f(hare))

        mu = 0
        tortoise = x0
        while tortoise != hare:
            tortoise = f(tortoise)
            hare = f(hare)   # Hare and tortoise move at same speed
            mu += 1

        lam = 1
        hare = f(tortoise)
        while tortoise != hare:
            hare = f(hare)
            lam += 1

        return lam, mu

    def brent(self, f, x0) -> (int, int):
        """Brent's cycle detection algorithm."""
        power = lam = 1
        tortoise = x0
        hare = f(x0)  # f(x0) is the element/node next to x0.
        while tortoise != hare:
            if power == lam:  # time to start a new power of two?
                tortoise = hare
                power *= 2
                lam = 0
            hare = f(hare)
            lam += 1

        tortoise = hare = x0
        for i in range(lam):
            hare = f(hare)

        mu = 0
        while tortoise != hare:
            tortoise = f(tortoise)
            hare = f(hare)
            mu += 1

        return lam, mu

    def find_largest_cycle(self, algorithm='floyd', samples: int = 10) -> int:
        """
        Find the largest cycle in the graph using the specified cycle detection algorithm.
        :param graph: The Graph object
        :param algorithm: The cycle detection algorithm to use ('floyd' or 'brent')
        :return: The length of the largest cycle (circumference)
        """
        vertices = list(self._graph.keys())
        if not vertices:
            return 0

        def get_next_vertex(v):
            neighbors = self.get_neighbors(v)
            if neighbors:
                return neighbors[0]
            return v

        largest_cycle = 0

        sampled_vertices = random.sample(vertices, min(samples, len(vertices)))

        for vertex in sampled_vertices:
            if algorithm == 'floyd':
                lam, _ = self.floyd(get_next_vertex, vertex)
            elif algorithm == 'brent':
                lam, _ = self.brent(get_next_vertex, vertex)
            else:
                raise ValueError("Unknown algorithm specified.")
            largest_cycle = max(largest_cycle, lam)

        return largest_cycle