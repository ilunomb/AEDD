import time
import matplotlib.pyplot as plt
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
# #Tamaño de la muestra para la estimación
sample_size = 100
num_vertices = 875713

# 1)

# largest_wcc_size = page_graph.largest_weakly_connected_component()
# num_wcc = page_graph.number_of_weakly_connected_components()

# print("Tamaño de la componente conexa más grande:", largest_wcc_size)
# print("Número total de componentes conexas:", num_wcc)


# 2)

# # Estimate time to calculate all shortest paths
# time_taken = page_graph.estimate_shortest_paths(samples=sample_size)
# # # time_taken = page_graph.floyd_warshall_partial(sample_size) # O(V^3)
# # # time_taken = page_graph.estimate_johnson_time(sample_size) # O(V^2 * E + V^2 * log(V))

# # Escalar el tiempo al tamaño completo del grafo
# estimated_time = time_taken * (num_vertices / sample_size)  # estimated time in seconds

# # Convertir el tiempo a horas, minutos y segundos
# hours, remainder = divmod(estimated_time, 3600)  # there are 3600 seconds in an hour
# minutes, seconds = divmod(remainder, 60)  # there are 60 seconds in a minute
# estimated_time_formatted = "{} hours, {} minutes, {} seconds".format(int(hours), int(minutes), int(seconds))
# print(f'Tiempo tardado en calcular para 100 nodos: {time_taken}')
# print("Tiempo estimado para calcular todos los caminos más cortos:", estimated_time_formatted)


# 3)

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


# 4)

# undir = page_graph.make_copy_graph_undirected()

# start = time.time()
# diameter = undir.estimate_diameter(samples=10)
# end = time.time()
# print("Diámetro estimado del grafo:", diameter)
# print("Tiempo de ejecución:", end - start, "segundos")


# 5)

# start = time.time()
# page_rank = page_graph.page_rank()
# top_page = sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:10]
# end = time.time()
# print("Tiempo de ejecución:", end - start, "segundos")
# for node, rank in top_page:
#     print(f"Node: {node}, PageRank: {rank}")


# 6)

# start = time.time()
# circumference = page_graph.largest_strongly_connected_component()
# n_scc = page_graph.number_of_strongly_connected_components()
# end = time.time()
# print("Circunferencia estimada del grafo:", circumference)
# print("Número de componentes fuertemente conexas:", n_scc)

# start = time.time()
# circumference, longest_cycle = page_graph.max_scc_cycle()
# end = time.time()
# print("Tiempo de ejecución:", end - start, "segundos")
# print("Circunferencia estimada del grafo:", circumference)
# print("Camino más largo:", longest_cycle)

# #make a function that checks if a list has repeated elements
# def has_duplicates(seq):
#     return len(seq) != len(set(seq))

# # #remove the last element of the longest cycle
# seq = longest_cycle[:-1]

# print(f'Tiene elementos repetidos: {has_duplicates(seq)}')

# def edges_exists(graph, cycle):
#     for i in range(len(cycle[:-1])):
#         if not graph.edge_exists(cycle[i], cycle[i+1]):
#             return False
#     return True

# print(f'Existen las aristas: {edges_exists(page_graph, longest_cycle)}')

# start = time.time()
# circumference = page_graph.estimate_circumference(samples=10)
# end = time.time()
# print("Circunferencia estimada del grafo:", circumference)
# print("Tiempo de ejecución:", end - start, "segundos")


# PUNTOS EXTRA


# 1)

# start = time.time()
# polygon_counts, polygons = page_graph.count_and_list_k_cycles(3)
# end = time.time()
# print("Número de triángulos en el grafo:", polygon_counts)
# print("Tiempo de ejecución:", end - start, "segundos")

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.bar(polygon_counts.keys(), polygon_counts.values())
# plt.xlabel('Number of sides')
# plt.ylabel('Number of polygons')
# plt.title('Number of polygons by number of sides')
# plt.xticks(range(3, 7))
# plt.grid(True)
# plt.show()


# 2)

# start = time.time()
# avarage_clustering_coefficient = page_graph.average_clustering_coefficient_undirected()
# end = time.time()
# print("Coeficiente de clustering promedio:", avarage_clustering_coefficient)
# print("Tiempo de ejecución:", end - start, "segundos")


# 3)

start = time.time()
max_vertex = page_graph.estimate_betweenness_centrality(samples=sample_size)
end = time.time()
print("Nodo con mayor centralidad de intermediación:", max_vertex)
print("Tiempo de ejecución:", end - start, "segundos")



#TESTING

# start = '143777'
# end = '576630'

# page_graph.print_shortest_path(start, end)

# dist, pred = page_graph.bfs_shortest_paths(start)

# print(f'Distancia: {dist[end]}')
# print(f'Camino: {pred[end]}')