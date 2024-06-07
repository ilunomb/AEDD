from typing import Optional, Any, List, Dict, Set


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
        # self._graph[vertex2]['neighbors'][vertex1] = data # Uncomment this line to make the graph undirected

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