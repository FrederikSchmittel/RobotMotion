import numpy as np
import matplotlib.pyplot as plt

class MotionModel:

    """
    The given code defines a class MotionModel that implements a robot localization 
    problem in a 2D discrete grid world. The robot can move forward (F), 
    backward (B), upwards(U) and downside(D) based on a given command. The motion model 
    for the robot is defined with three variable probabilities:
    CORRECT_DIRECTION_PROBABILITY: 50% chance that the robot moves in the correct 
    direction (i.e. F moves the robot forward and B moves the robot backward).
    NO_MOVEMENT_PROBABILITY: 10% chance that the robot does not move or 
    cannot execute the command.
    OPPOSITE_DIRECTION_PROBABILITY: 40% chance that the robot moves in the 
    opposite direction (i.e. F moves the robot backward and B moves the robot forward).

    The class takes in the world size, initial position of the robot, and the number of 
    actions the robot can take. The motion_model method of the class calculates the new 
    belief state of the agent after taking a given action, based on the motion model. The 
    random_actions method of the class generates random actions for the robot. The 
    plot_belief method of the class plots the belief matrix as a 2D image and a 3D surface 
    plot. The belief matrix is a 2D NumPy array where each element represents a probability value.

    In the motion_model method, the new belief state of the agent is calculated by iterating 
    over the entire grid and checking for three conditions for each cell:
    - If the robot moves in the correct direction, the new position is updated with the probability of CORRECT_DIRECTION_PROBABILITY.
    - If there is no movement, the current position is updated with the probability of NO_MOVEMENT_PROBABILITY.
    - If the robot moves in the opposite direction, the opposite position is updated with the probability of OPPOSITE_DIRECTION_PROBABILITY.

    The plot_belief method displays the belief matrix as a 2D image and a 3D surface plot in separate subplots of a single figure. 
    The imshow and plot_surface methods of the Matplotlib library are used to plot the 2D image and 3D surface plot, respectively.

    Args:
        world_size (int): The size of the 2D square grid world.
        initial_position (tuple): The initial position of the robot in the form (x, y).
        num_actions (int): The number of actions the robot can take.

    Returns:
        None

    Example:
    world_size = 30
    initial_position = [10, 14]
    num_actions = 500
    model = MotionModel(world_size, initial_position, num_actions)
    model.run_motion_model()
    """

    # The probability that the robot moves in the correct direction according to the action taken.
    CORRECT_DIRECTION_PROBABILITY: float = 0.5

    # The probability that the robot does not move and stays in its current position.
    NO_MOVEMENT_PROBABILITY: float = 0.1

    # The probability that the robot moves in the opposite direction of the intended action.
    OPPOSITE_DIRECTION_PROBABILITY: float = 0.4

    def __init__(self, world_size: int, initial_position: tuple[int, int], num_actions: int) -> None:
        """
        Initialize the robot localization problem with the 
        given world size, initial position, and number of actions.

        Args:
            world_size (int): The size of the 2D square grid world.
            initial_position (tuple): The initial position 
            of the robot in the form (x, y).
            num_actions (int): The number of actions the robot can take.

        Returns:
            None
        """

        self.world_size: int = world_size
        self.actions: np.ndarray = self.random_actions(num_actions)
        self.belief: np.ndarray = np.zeros([world_size, world_size])
        self.belief[initial_position[0], initial_position[1]]: float = 1.0
        self.num_actions: int = num_actions

    def motion_model(self, action: np.ndarray) -> None:
        """
        Calculates the new belief state of the agent after 
        taking a given action, based on a motion model.

        The motion model is defined as follows:
        - With a probability of CORRECT_DIRECTION_PROBABILITY, 
        the agent moves in the correct direction specified by the action.
        - With a probability of NO_MOVEMENT_PROBABILITY, 
        the agent does not move at all.
        - With a probability of OPPOSITE_DIRECTION_PROBABILITY, 
        the agent moves in the opposite direction specified by the action.

        Args:
            action: A NumPy array of shape (,2) representing 
            the direction of movement. The first element specifies 
            the change in the x direction, and the second element 
            specifies the change in the y direction.

        Returns:
            None. The new belief state is stored in the `belief` 
            attribute of the `MotionModel` instance.
        """

        new_belief: np.ndarray = np.zeros([self.world_size, self.world_size])

        for i in range(self.world_size):
            for j in range(self.world_size):
                if i + action[0] >= 0 and j + action[1] >= 0 and i + action[0] < self.world_size and j + action[1] < self.world_size:

                    # Move in the correct direction
                    new_pos = np.array([i, j]) + action
                    new_pos = np.mod(new_pos, self.world_size)
                    new_belief[int(new_pos[0]), int(
                        new_pos[1])] += self.CORRECT_DIRECTION_PROBABILITY * self.belief[i, j]

                    # No movement
                    new_belief[i, j] += self.NO_MOVEMENT_PROBABILITY * \
                        self.belief[i, j]

                    # Move in the opposite direction
                    opp_pos = np.array([i, j]) - action
                    opp_pos = np.mod(opp_pos, self.world_size)
                    new_belief[int(opp_pos[0]), int(
                        opp_pos[1])] += self.OPPOSITE_DIRECTION_PROBABILITY * self.belief[i, j]

        # Normalize the belief state and store it in the object attribute
        if np.sum(new_belief) > 0:
            self.belief = new_belief/np.sum(new_belief)

    def plot_belief(self) -> None:
        """
        Plots the belief matrix as an image and a 3D surface plot.

        Displays the belief matrix as a 2D image and a 3D surface 
        plot in separate subplots of a single figure.
        The belief matrix is assumed to be a 2D numpy array 
        where each element represents a probability value.
        The first subplot shows the belief matrix as a 2D 
        image using a "hot" colormap and no axis labels.
        The second subplot displays the belief matrix as a 
        3D surface plot with the x and y coordinates of each
        probability value obtained from the indices of the 
        belief matrix, and the z coordinate corresponding to
        the probability value itself. The surface plot is 
        colored using a "viridis" colormap and has x, y, 
        and z axis labels.

        Args:
            None

        Returns:
            None
        """

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 6))

        # Plot the belief matrix as an image
        ax1.imshow(self.belief, cmap='hot', interpolation='nearest')
        ax1.set_title("Belief Matrix")
        ax1.axis('off')

        # creates a 3D surface using the coordinates X and Y, where the Z-coordinates are determined by the values in self.belief, and adds axis labels.
        x = range(self.belief.shape[0])
        y = range(self.belief.shape[1])
        X, Y = np.meshgrid(x, y)

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(X, Y, self.belief, cmap='viridis')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Belief')

        plt.show()

    def random_actions(self, number_of_actions: int) -> np.ndarray:
        """
        Generates n random vectors with values of -1, 0, or 1 for 
        both dimensions.

        Args:
            n: An integer specifying the number of random vectors to 
            generate.

        Returns:
            A NumPy array of shape (n, 2) representing a collection 
            of random vectors with x and y components which defines the 
            direction of movement.
        """
        # Initialize an empty NumPy array to store the random vectors
        random_actions = np.empty([number_of_actions, 2])

        # Generate n random vectors and store them in the array
        for i in range(number_of_actions):
            rand_x = np.random.randint(-1, 2)
            rand_y = np.random.randint(-1, 2)

            # If both values are 0, generate new values until they are not both 0
            while rand_x == 0 and rand_y == 0:
                rand_x = np.random.randint(-1, 2)
                rand_y = np.random.randint(-1, 2)

            # Store the generated vector in the array
            random_actions[i, 0] = rand_x
            random_actions[i, 1] = rand_y

        # Return the array of random vectors
        return random_actions

    def run_motion_model(self) -> np.ndarray:
        """
        Applies the motion model to update the belief state 
        for each action in the list of actions, and plots the 
        final belief state.

        Returns:
            A NumPy array of shape (n, m) representing the 
            final belief state, where n is the number of 
            rows and m is the number of columns in the environment.
        """

        for i in range(self.num_actions):
            self.motion_model(self.actions[i])
        self.plot_belief()
