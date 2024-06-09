import random
import time
import numpy as np
from typing import Optional, Any, List, Dict, Set, Tuple


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
            raise ValueError("The vertexes do not exist")
        self._graph[vertex1]['neighbors'][vertex2] = data

    def get_neighbors(self, vertex) -> List[str]:
        """
        Get the list of vertex neighbors
        :param vertex: the vertex to query
        :return: the list of neighbor vertexes
        """
        if vertex in self._graph:
            return list(self._graph[vertex]['neighbors'].keys())
        else:
            return []

    def get_vertex_data(self, vertex: str) -> Optional[Any]:
        """
        Gets  vertex associated data
        :param vertex: the vertex name
        :return: the vertex data
        """
        if self.vertex_exists(vertex):
            return self._graph[vertex]['data']
        else:
            return None

    def get_edge_data(self, vertex1: str, vertex2: str) -> Optional[Any]:
        """
        Gets the vertexes edge data
        :param vertex1: the vertex1 name
        :param vertex2: the vertex2 name
        :return: vertexes edge data
        """
        if self.edge_exists(vertex1, vertex2):
            return self._graph[vertex1]['neighbors'][vertex2]
        raise ValueError("The edge does not exist")

    def print_graph(self) -> None:
        """
        Prints the graph
        """
        for vertex, data in self._graph.items():
            print("Vertex:", vertex)
            print("Data:", data['data'])
            print("Neighbors:", data['neighbors'])
            print("")

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
    
    def _strong_connect(self, vertex: str) -> None:
        """
        Non-recursive helper function for Tarjan's algorithm to find strongly connected components
        """
        stack = [(vertex, 0)]
        visited = set()
        
        while stack:
            v, index = stack[-1]
            
            if v not in visited:
                visited.add(v)
                self._indices[v] = self._index
                self._low_links[v] = self._index
                self._index += 1
                self._stack.append(v)
                self._on_stack.add(v)
            
            neighbors = self.get_neighbors(v)
            
            if index < len(neighbors):
                neighbor = neighbors[index]
                stack[-1] = (v, index + 1)
                
                if neighbor not in self._indices:
                    stack.append((neighbor, 0))
                elif neighbor in self._on_stack:
                    self._low_links[v] = min(self._low_links[v], self._indices[neighbor])
            else:
                if self._low_links[v] == self._indices[v]:
                    scc = set()
                    while True:
                        w = self._stack.pop()
                        self._on_stack.remove(w)
                        scc.add(w)
                        if w == v:
                            break
                    self._sccs.append(scc)
                stack.pop()
                if stack:
                    w, _ = stack[-1]
                    self._low_links[w] = min(self._low_links[w], self._low_links[v])

    def find_strongly_connected_components(self) -> List[Set[str]]:
        """
        Finds and returns all strongly connected components
        """
        self._index = 0
        self._stack = []
        self._indices = {}
        self._low_links = {}
        self._on_stack = set()
        self._sccs = []

        for vertex in self._graph:
            if vertex not in self._indices:
                self._strong_connect(vertex)

        return self._sccs

    def largest_strongly_connected_component(self) -> int:
        """
        Returns the size of the largest strongly connected component
        """
        sccs = self.find_strongly_connected_components()
        return max(len(scc) for scc in sccs) if sccs else 0

    def number_of_strongly_connected_components(self) -> int:
        """
        Returns the number of strongly connected components
        """
        sccs = self.find_strongly_connected_components()
        return len(sccs)
    
    def floyd_warshall_partial(self, sample_size: int) -> float:
        vertices = list(self._graph.keys())
        sampled_vertices = random.sample(vertices, sample_size)
        dist = {v: {u: float('inf') for u in sampled_vertices} for v in sampled_vertices}

        for v in sampled_vertices:
            dist[v][v] = 0

        for v in sampled_vertices:
            for u, weight in self._graph[v]['neighbors'].items():
                if u in sampled_vertices:
                    dist[v][u] = weight

        start_time = time.time()

        for k in sampled_vertices:
            for i in sampled_vertices:
                for j in sampled_vertices:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        end_time = time.time()
        return end_time - start_time

    # def _initialize_single_source(self, source):
    #     dist = {vertex: float('inf') for vertex in self._graph}
    #     dist[source] = 0
    #     return dist

    # def _relax(self, u, v, weight, dist):
    #     if dist[u] + weight < dist[v]:
    #         dist[v] = dist[u] + weight

    # def bellman_ford(self, source):
    #     dist = self._initialize_single_source(source)
    #     for _ in range(len(self._graph) - 1):
    #         for u in self._graph:
    #             for v, weight in self._graph[u]['neighbors'].items():
    #                 self._relax(u, v, weight, dist)
    #     for u in self._graph:
    #         for v, weight in self._graph[u]['neighbors'].items():
    #             if dist[u] + weight < dist[v]:
    #                 raise ValueError("Graph contains a negative-weight cycle")
    #     return dist

    # def dijkstra(self, source):
    #     dist = self._initialize_single_source(source)
    #     pq = [(0, source)]
    #     while pq:
    #         current_dist, u = heapq.heappop(pq)
    #         if current_dist > dist[u]:
    #             continue
    #         for v, weight in self._graph[u]['neighbors'].items():
    #             if dist[u] + weight < dist[v]:
    #                 dist[v] = dist[u] + weight
    #                 heapq.heappush(pq, (dist[v], v))
    #     return dist

    # def johnson(self):
    #     # Step 1: Add a new vertex s and connect it to all other vertices with edge weight 0
    #     s = 's'
    #     self.add_vertex(s)
    #     for vertex in self._graph:
    #         if vertex != s:
    #             self.add_edge(s, vertex, 0)

    #     # Step 2: Run Bellman-Ford from the new vertex s
    #     try:
    #         h = self.bellman_ford(s)
    #     except ValueError:
    #         print("Graph contains a negative-weight cycle")
    #         return

    #     # Step 3: Remove the added vertex s
    #     del self._graph[s]

    #     # Step 4: Reweight all edges
    #     for u in self._graph:
    #         for v in self._graph[u]['neighbors']:
    #             self._graph[u]['neighbors'][v] += h[u] - h[v]

    #     # Step 5: Run Dijkstra's algorithm for each vertex
    #     all_pairs_dist = {}
    #     for u in self._graph:
    #         all_pairs_dist[u] = self.dijkstra(u)

    #     # Step 6: Reverse the reweighting
    #     for u in all_pairs_dist:
    #         for v in all_pairs_dist[u]:
    #             if all_pairs_dist[u][v] != float('inf'):
    #                 all_pairs_dist[u][v] += h[v] - h[u]

    #     return all_pairs_dist

    # def estimate_johnson_time(self, sample_size: int):
    #     vertices = list(self._graph.keys())
    #     sampled_vertices = random.sample(vertices, sample_size)

    #     # Create a subgraph with the sampled vertices
    #     subgraph = Graph()
    #     for v in sampled_vertices:
    #         subgraph.add_vertex(v)
    #     for u in sampled_vertices:
    #         for v in self._graph[u]['neighbors']:
    #             if v in sampled_vertices:
    #                 subgraph.add_edge(u, v, self._graph[u]['neighbors'][v])

    #     start_time = time.time()
    #     subgraph.johnson()
    #     end_time = time.time()

    #     time_taken = end_time - start_time
    #     estimated_time = time_taken * (len(self._graph) / sample_size) ** 3

    #     return estimated_time
    
    # def count_cycle_triangles(self):
    #     """
    #     Counts the number of triangles in the graph using the cycle triangle counting method
    #     """
    #     count = 0
    #     for vertex in self._graph:
    #         neighbors = self.get_neighbors(vertex)
    #         for i in range(len(neighbors)):
    #             for j in range(i + 1, len(neighbors)):
    #                 if self.edge_exists(neighbors[i], neighbors[j]):
    #                     count += 1
    #     return count

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

        for u in self._graph:
            for v in self.get_neighbors(u):
                if u < v:
                    S = set(GT.get_neighbors(u)).intersection(self.get_neighbors(v))
                    for w in S:
                        if u < w:
                            c += 1

        return c