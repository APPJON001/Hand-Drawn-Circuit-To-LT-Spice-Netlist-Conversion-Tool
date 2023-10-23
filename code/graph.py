import networkx as nx
import numpy as np
from PIL import Image


def find_terminals_graph(image_path, terminal_locations):
    # Load your binary edge-detected image
    binary_image = Image.open(image_path)
    binary_array = np.array(binary_image)

    # Create a graph
    G = nx.Graph()

    # Iterate through the image and add nodes and edges to the graph
    height, width = binary_array.shape
    for x in range(width):
        for y in range(height):
            if binary_array[y, x] == 0:  # Check if the pixel is black
                G.add_node((x, y))
                # Check adjacent pixels and add edges
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        new_x, new_y = x + dx, y + dy
                        if 0 <= new_x < width and 0 <= new_y < height and binary_array[new_y, new_x] == 0:
                            G.add_edge((x, y), (new_x, new_y))

    # Find connected components
    connected_components = list(nx.connected_components(G))
    updated_connected_terminals = []

    # Get closest graph-connected pixels to existing terminals to make new terminals to check connections
    for terminal in terminal_locations:
        closest_pixel = min(G.nodes, key=lambda pixel: ((pixel[0] - terminal[0]) ** 2 + (pixel[1] - terminal[1]) ** 2) ** 0.5)
        updated_connected_terminals.append(closest_pixel)

    print("updated terminals = ", updated_connected_terminals)

    # Create a dictionary to store connections between terminal locations
    connections = {tuple(terminal): [] for terminal in updated_connected_terminals}

    # Check connections for each updated closest terminal pixel
    for terminal in updated_connected_terminals:
        # Find the connected component that contains this pixel
        for component in connected_components:
            if terminal in component:
                # Iterate through terminal locations to find others in the same component
                for other_terminal in updated_connected_terminals:
                    if other_terminal != terminal:
                        if tuple(other_terminal) in component:
                            connections[tuple(terminal)].append(tuple(other_terminal))

    # Check if there are exactly two unlinked terminals, and if so, link them to each other
    unlinked_terminals = [terminal for terminal, connected_terminals in connections.items() if not connected_terminals]
    if len(unlinked_terminals) == 2:
        terminal1, terminal2 = unlinked_terminals
        connections[terminal1].append(terminal2)
        connections[terminal2].append(terminal1)

    # Print the connections for each terminal
    #for terminal, connected_terminals in connections.items():
    #   print(f"Terminal {terminal} is connected to {connected_terminals}")

    return connections

