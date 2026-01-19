import random
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
place_name = "名古屋工業大学, 愛知県, 日本"

G = ox.graph_from_place(place_name, network_type="walk")
fig, ax = ox.plot_graph(G)

plt.show()

