from collections import defaultdict, deque
import random
import time
from typing import Optional, Any, List, Dict, Set, Tuple, Generator
from tqdm import tqdm
from multiprocessing import Process, Manager
import matplotlib.pyplot as plt

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

    def print_shortest_path(self, start: str, end: str) -> None:
        """
        Print the shortest path from start to end
        :param start: The start vertex
        :param end: The end vertex
        """
        if not self.vertex_exists(start) or not self.vertex_exists(end):
            print("One or both vertices not found in the graph")
            return
        
        par, dist = self.bfs(start)
        
        if dist[end] == float('inf'):
            print("Source and Destination are not connected")
            return
        
        path = []
        current_node = end
        while current_node is not None:
            path.append(current_node)
            current_node = par[current_node]
        
        path.reverse()
        print(" -> ".join(path))

    def estimate_shortest_paths(self, samples: int) -> float:
        vertices = list(self._graph.keys())
        sampled_vertices = random.sample(vertices, samples)

        start_time = time.time()
        for vertex in tqdm(sampled_vertices, desc="Estimating shortest paths", unit="vertex"):
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

        for _ in range(samples): # takes into account not analyzing only one connected component
        
            vertex = random.choice(vertices)

            for _ in tqdm(range(100), desc="Estimating diameter", unit="iteration"):
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

        for iteration in tqdm(range(max_iterations), desc="PageRank", unit="iteration"):
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

        for u in tqdm(self._graph, desc="Counting cycle triangles", unit="vertex"):
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
        for vertex in tqdm(undirected_graph._graph, desc="Counting triangles", unit="vertex"):
            neighbors = undirected_graph.get_neighbors(vertex)
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if undirected_graph.edge_exists(neighbors[i], neighbors[j]):
                        count += 1
        return count // 3           

    def has_k_cycle_util(self, v: str, visited: set, recStack: dict, k: int):
        stack = [(v, [v])]
        while stack:
            (node, path) = stack.pop()
            if node not in visited:
                visited.add(node)
                recStack[node] = path
                for neighbour in self.get_neighbors(node):
                    if neighbour not in visited:
                        stack.append((neighbour, path + [neighbour]))
                    elif neighbour in recStack and neighbour in path:
                        cycle_path = path
                        if len(cycle_path) - cycle_path.index(neighbour) == k:
                            return cycle_path[cycle_path.index(neighbour):]
            elif node in recStack and node in path:
                if len(path) - path.index(node) == k:
                    return path[path.index(node):]
                else:
                    return []
        return []

    def has_k_cycle(self, k=3, timeout=30):
        recStack = {}
        start = time.time()
        for node in tqdm(self._graph, desc="Checking for cycles of length " + str(k)):
            if time.time() - start > timeout:
                break
            visited = set()  # reset visited set for each node
            cycle = self.has_k_cycle_util(node, visited, recStack, k=k)
            if cycle:
                return cycle
        return []

    def worker(self, k):
        cycle = self.has_k_cycle(k)
        if cycle:
            return len(cycle)
        return 0

    def circumference(self):
        vertices = list(self._graph.keys())
        circumference = [0]
    
        def binary_search_time(low, high):
            if low > high:
                return
            mid = (low + high) // 2
            cycle_length = self.worker(mid)
            circumference[0] = max(circumference[0], cycle_length)
            if cycle_length == mid:
                binary_search_time(mid + 1, high)
            else:
                binary_search_time(low, mid - 1)
    
        binary_search_time(2, len(vertices))
        return circumference[0]
    
    def average_clustering_coefficient_undirected(self) -> float:
        """
        Calculate the average clustering coefficient of the undirected graph
        """
        clustering_coefficient = 0
        undirected_graph = self.make_copy_graph_undirected()

        computed_neighbors = {vertex: set(undirected_graph.get_neighbors(vertex)) for vertex in tqdm(undirected_graph._graph, desc="Compting neighbors", unit="vertex")}

        for vertex in tqdm(undirected_graph._graph, desc="Calculating clustering coefficient", unit="vertex"):
            neighbors = computed_neighbors[vertex]
            count = 0
            for neighbor in neighbors:
                count += len(set(undirected_graph.get_neighbors(neighbor)).intersection(neighbors))
            clustering_coefficient += count / (len(neighbors) * (len(neighbors) - 1)) if len(neighbors) > 1 else 0
        return clustering_coefficient / len(undirected_graph._graph)

    def average_clustering_coefficient_directed(self) -> float:
        """
        Calculate the average clustering coefficient of the directed graph
        """
        clustering_coefficient = 0
        computed_neighbors = {vertex: set(self.get_neighbors(vertex)) for vertex in tqdm(self._graph, desc="Compting neighbors", unit="vertex")}

        for vertex in tqdm(self._graph, desc="Calculating clustering coefficient", unit="vertex"):
            neighbors = computed_neighbors[vertex]
            count = 0
            for neighbor in neighbors:
                count += len(set(self.get_neighbors(neighbor)).intersection(neighbors))
            clustering_coefficient += count / (len(neighbors) * (len(neighbors) - 1)) if len(neighbors) > 1 else 0
        return clustering_coefficient / len(self._graph)


    def bfs_bc_helper(self, start: str) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
        """
        Perform BFS and return the shortest paths and predecessor lists.
        :param start: The start vertex
        :return: distance dictionary and predecessor lists
        """
        dist = {v: float('inf') for v in self._graph}
        pred = {v: [] for v in self._graph}
        
        dist[start] = 0
        q = deque([start])
        
        while q:
            node = q.popleft()
            for neighbor in self.get_neighbors(node):
                if dist[neighbor] == float('inf'):
                    dist[neighbor] = dist[node] + 1
                    q.append(neighbor)
                if dist[neighbor] == dist[node] + 1:
                    pred[neighbor].append(node)
                    
        return dist, pred

    def estimate_betweenness_centrality(self, samples: int = 10) -> str:
        """
        Estimate the vertex with the highest betweenness centrality.
        :param samples: The number of vertices to sample
        :return: The vertex with the highest estimated betweenness centrality
        """
        vertices = list(self._graph.keys())
        sampled_vertices = random.sample(vertices, samples)
        
        betweenness = defaultdict(float)
        
        for vertex in tqdm(sampled_vertices):
            # Get shortest paths and predecessor lists from the current vertex
            dist, pred = self.bfs_bc_helper(vertex)
            
            # Initialize the dependencies
            delta = {v: 0 for v in self._graph}
            
            # Perform accumulation of dependencies in reverse order
            stack = [v for v in vertices if dist[v] < float('inf')]
            stack.sort(key=lambda v: -dist[v])
            
            for w in stack:
                coeff = (1 + delta[w]) / len(pred[w]) if pred[w] else 1
                for v in pred[w]:
                    delta[v] += dist[v] / dist[w] * coeff
                if w != vertex:
                    betweenness[w] += delta[w]
        
        # Find the vertex with the highest betweenness centrality
        max_vertex = max(betweenness, key=betweenness.get)
        return max_vertex
    

    def find_k_cycles(self, start_node, k):
        """Find all k-cycles starting from a given node."""
        transposed = self.transpose()
        stack = [(start_node, [start_node], 1)]
        k_cycles = []

        while stack:
            current_node, path, current_depth = stack.pop()

            if current_depth == k:
                if start_node in set(self.get_neighbors(current_node)) | set(transposed.get_neighbors(current_node)):
                    k_cycles.append(path)
            elif current_depth < k:
                for neighbor in set(self.get_neighbors(current_node)) | set(transposed.get_neighbors(current_node)):
                    if neighbor not in path:
                        stack.append((neighbor, path + [neighbor], current_depth + 1))

        return k_cycles

    def estimate_k_cycles(self, k, n):
        """Estimate the number of k-cycles in the graph by sampling n nodes."""
        total_k_polygons = 0
        vertices = list(self._graph.keys())
        random.shuffle(vertices)

        for i in range(n):
            start_node = vertices[i]
            k_cycles = self.find_k_cycles(start_node, k)
            total_k_polygons += len(k_cycles)

        return total_k_polygons / (2 * k) / n * len(vertices)

    def plot_polygons(self, sides_range, tries_range):
        polygons_count = [self.estimate_k_cycles(k, tries) for k, tries in tqdm(zip(sides_range, tries_range))]

        plt.bar(sides_range, polygons_count)
        plt.yscale('log')
        plt.xlabel('Number of sides')
        plt.ylabel('Number of polygons')
        plt.title('Number of polygons by number of sides')
        plt.show()