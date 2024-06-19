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


# # 2)

# # Estimate time to calculate all shortest paths
# time_taken = page_graph.estimate_shortest_paths(samples=sample_size)

# # Escalar el tiempo al tamaño completo del grafo
# estimated_time = time_taken * (num_vertices / sample_size)  # estimated time in seconds

# # Convertir el tiempo a horas, minutos y segundos
# hours, remainder = divmod(estimated_time, 3600)  # there are 3600 seconds in an hour
# minutes, seconds = divmod(remainder, 60)  # there are 60 seconds in a minute
# estimated_time_formatted = "{} hours, {} minutes, {} seconds".format(int(hours), int(minutes), int(seconds))
# print(f'Tiempo tardado en calcular para 100 nodos: {time_taken}')
# print("Tiempo estimado para calcular todos los caminos más cortos:", estimated_time_formatted)


# # 3)

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


# # 4)

# undir = page_graph.make_copy_graph_undirected()

# start = time.time()
# diameter = undir.estimate_diameter(samples=10)
# end = time.time()
# print("Diámetro estimado del grafo (undirected):", diameter)
# print("Tiempo de ejecución:", end - start, "segundos")

# start = time.time()
# diameter = page_graph.estimate_diameter(samples=10)
# end = time.time()
# print("Diámetro estimado del grafo (directed):", diameter)
# print("Tiempo de ejecución:", end - start, "segundos")


# # 5)

# start = time.time()
# page_rank = page_graph.page_rank()
# top_page = sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:10]
# end = time.time()
# print("Tiempo de ejecución:", end - start, "segundos")
# for node, rank in top_page:
#     print(f"Node: {node}, PageRank: {rank}")


# # 6)

# start = time.time()
# circumference = page_graph.circumference()
# end = time.time()
# print("Circunferencia estimada del grafo:", circumference)
# print("Tiempo de ejecución:", end - start, "segundos")



# PUNTOS EXTRA


# 1)

sides_range = range(3, 6) 

# Cantidad de iteraciones para estimar el número de polígonos. A mayor cantidad, más precisa la estimación, pero a la vez tarda más tiempo.
tries_range = [100, 50, 25]      

page_graph.plot_polygons(sides_range, tries_range)


# 2)

# start = time.time()
# avarage_clustering_coefficient = page_graph.average_clustering_coefficient_undirected()
# end = time.time()
# print("Coeficiente de clustering promedio (undirected):", avarage_clustering_coefficient)
# print("Tiempo de ejecución:", end - start, "segundos")

# start = time.time()
# avarage_clustering_coefficient = page_graph.average_clustering_coefficient_directed()
# end = time.time()
# print("Coeficiente de clustering promedio (directed):", avarage_clustering_coefficient)
# print("Tiempo de ejecución:", end - start, "segundos")

# 3)

# start = time.time()
# max_vertex = page_graph.estimate_betweenness_centrality(samples=sample_size)
# end = time.time()
# print("Nodo con mayor centralidad de intermediación:", max_vertex)
# print("Tiempo de ejecución:", end - start, "segundos")