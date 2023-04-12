import numpy as np
from noise import pnoise2


class World:
    """
    A class representing a world generated with Perlin noise.

    Attributes:
        size (int): Size of the world.
        world (np.ndarray): A 2D NumPy array representing the generated world.

    Methods:
        __init__(self, size: int) -> None:
            Initializes the `World` class instance with the given size.
        create_perlin_noise(self, scale: float = 1000.0, 
                                octaves: int = 5, 
                                persistence: float = 100.0, 
                                lacunarity: float = 2.0) -> None:
            Generates Perlin noise and saves it in the `World` instance.
        world_to_array(self) -> np.ndarray:
            Converts the `World` instance to a NumPy array.

    Example:
    world_size = 1000
    world = World(world_size)       # create a new world with a size of world_size x world_size
    
    world.create_perlin_noise()    # generate the world using Perlin noise

    arr = world.world_to_array()   # convert the world to a NumPy array

    plt.imshow(arr, cmap='binary', interpolation='nearest')
    plt.show()

    """

    def __init__(self, size: int) -> None:
        """
        Initializes the `World` class instance with the given size.

        Args:
            size (int): Size of the world.
        """
        self.size = size
        self.world = np.zeros((size, size))

    def create_perlin_noise(self, scale: float = 1000.0,
                            octaves: int = 5,
                            persistence: float = 100.0,
                            lacunarity: float = 2.0) -> None:
        """
        Generates Perlin noise and saves it in the `World` instance.

        Args:
            scale (float): Scale of the noise in pixels. A larger
            value leads to larger patterns. Default is 1000.0.
            octaves (int): Number of noise layers to be stacked on
            top of each other to generate the final pattern. Each layer 
            has a higher frequency (smaller scale) and amplitude (larger
            size) than the previous layer. Default is 5.
            persistence (float): Controls how fast the amplitude will
            drop off for each octave. A higher value will result in a
            more abrupt drop-off and a "rougher" pattern. Default is 100.
            lacunarity (float): Controls how different the scale frequency
            between octaves is. A higher value will result in larger differences
            between the scale frequencies, leading to a "rougher" pattern. Default is 2.0.
        """
        # The `noise` library provides methods for
        # generating various types of noise patterns.
        # Here we are using the `pnoise2` method
        # which generates Perlin noise in 2 dimensions.
        # For more information about the `noise`
        # library and its methods, visit https://pypi.org/project/noise/.

        for x in range(self.size):
            for y in range(self.size):
                noise = pnoise2(x / scale, y / scale, octaves=octaves,
                                persistence=persistence, lacunarity=lacunarity)
                if noise > 0.2:
                    self.world[x, y] = 1
                else:
                    self.world[x, y] = 0

    def world_to_array(self) -> np.ndarray:
        """
        Converts the `World` instance to a NumPy array.

        Returns:
            np.ndarray: A NumPy array representing the `World` instance.
        """
        return np.array(self.world)
