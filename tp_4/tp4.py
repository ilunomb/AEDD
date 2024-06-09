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

# Calcular y mostrar los resultados
largest_scc_size = page_graph.largest_strongly_connected_component()
num_scc = page_graph.number_of_strongly_connected_components()

print("Tamaño de la componente fuertemente conexa más grande:", largest_scc_size)
print("Número total de componentes fuertemente conexas:", num_scc)

#Tamaño de la muestra para la estimación
sample_size = 100
time_taken = page_graph.floyd_warshall_partial(sample_size) # O(V^3)
# time_taken = page_graph.estimate_johnson_time(sample_size) # O(V^2 * E + V^2 * log(V))


# Escalar el tiempo al tamaño completo del grafo
num_vertices = 875713
estimated_time = time_taken * (num_vertices / sample_size) ** 3  # estimated time in seconds

years, remainder = divmod(estimated_time, 31536000)  # there are 31536000 seconds in a year
days, remainder = divmod(remainder, 86400)  # there are 86400 seconds in a day
hours, remainder = divmod(remainder, 3600)  # there are 3600 seconds in an hour
minutes, seconds = divmod(remainder, 60)  # there are 60 seconds in a minute

estimated_time_formatted = "{} years, {} days, {} hours, {} minutes, {} seconds".format(int(years), int(days), int(hours), int(minutes), int(seconds))

print(f"It will take approximately {estimated_time_formatted} for the complete graph.")

num_triangles = page_graph.count_and_list_cycle_triangles()
print("Número de triángulos en el grafo:", num_triangles)