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