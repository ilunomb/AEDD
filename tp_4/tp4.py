import time
from graph import Graph

page_graph = Graph()

with open('tp_4\web-Google.txt', 'r') as file:
    for l in file:
        if "# FromNodeId	ToNodeId" in l:
            break
    for l in file:
        if not l:
            break
        edge = tuple(int(v.replace("\n", "").replace("\t", "")) for v in l.split("\t"))
        for v in edge:
            if not page_graph.vertex_exists(v):
                page_graph.add_vertex(str(v))
        page_graph.add_edge(str(edge[0]), str(edge[1]))

# # Calcular y mostrar los resultados
# largest_wcc_size = page_graph.largest_weakly_connected_component()
# num_wcc = page_graph.number_of_weakly_connected_components()

# print("Tamaño de la componente conexa más grande:", largest_wcc_size)
# print("Número total de componentes conexas:", num_wcc)

# #Tamaño de la muestra para la estimación
sample_size = 100
# # Estimate time to calculate all shortest paths
# time_taken = page_graph.estimate_shortest_paths(samples=sample_size)
# # time_taken = page_graph.floyd_warshall_partial(sample_size) # O(V^3)
# # time_taken = page_graph.estimate_johnson_time(sample_size) # O(V^2 * E + V^2 * log(V))

# # Escalar el tiempo al tamaño completo del grafo
# num_vertices = 875713
# estimated_time = time_taken * (num_vertices / sample_size)  # estimated time in seconds

# hours, remainder = divmod(estimated_time, 3600)  # there are 3600 seconds in an hour
# minutes, seconds = divmod(remainder, 60)  # there are 60 seconds in a minute
# estimated_time_formatted = "{} hours, {} minutes, {} seconds".format(int(hours), int(minutes), int(seconds))
# print(estimated_time_formatted)

# start = time.time()
# num_triangles = page_graph.count_triangles()
# end = time.time()
# print("Número total de triángulos en el grafo:", num_triangles)
# print("Tiempo de ejecución:", end - start, "segundos")

# start = time.time()
# num_triangles_cycle, triangles = page_graph.count_and_list_cycle_triangles()
# end = time.time()
# print("Número de triángulos cycle en el grafo:", num_triangles_cycle)
# print("Tiempo de ejecución:", end - start, "segundos")

# start = time.time()
# diameter = page_graph.estimate_diameter(samples=10)
# end = time.time()
# print("Diámetro estimado del grafo:", diameter)
# print("Tiempo de ejecución:", end - start, "segundos")

# start = time.time()
# page_rank = page_graph.page_rank()
# top_page = sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:10]
# end = time.time()
# print("Tiempo de ejecución:", end - start, "segundos")
# for node, rank in top_page:
#     print(f"Node: {node}, PageRank: {rank}")

# start = time.time()
# circumference = page_graph.largest_strongly_connected_component()
# n_scc = page_graph.number_of_strongly_connected_components()
# end = time.time()
# print("Circunferencia estimada del grafo:", circumference)
# print("Número de componentes fuertemente conexas:", n_scc)

undir = page_graph.make_copy_graph_undirected()

start = '143777'

end = '576630'

par, dist = undir.bfs(start)

lenght_to = dist['576630']

path = par['576630']

while path != start:
    print(path)
    path = par[path]

print(lenght_to)