import os
import numpy as np
from PIL import Image

class MapInterface:
    def __init__(self, map_folder):
        self.map_folder = map_folder
        self.map_data = None

    def load_map(self, map_name):
        map_file = os.path.join(self.map_folder, map_name + ".pgm")

        image = Image.open(map_file)
        self.map_data = np.array(image)>230

# Example usage
map_folder = "map"
map_interface = MapInterface(map_folder)
map_name = "sk2_0424"
map_interface.load_map(map_name)

# save the map as an image
import matplotlib.pyplot as plt
plt.imsave(f"map/{map_name}.png", map_interface.map_data, cmap="gray")