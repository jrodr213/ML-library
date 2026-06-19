from Node import Node

class Base:
    """
    Shared parent class for network types built from Node objects.
    """

    def __init__(self, input_size, learning_rate, output_size=None):
        """
        Stores the common network configuration and creates its nodes.
        """
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.output_size = input_size if output_size is None else output_size
        self.nodes = []
        self.set_up()

    def set_up(self):
        """
        Creates the node objects used by the network.
        """
        raise NotImplementedError("subclasses must implement set_up")

    def run(self, input):
        """
        Sends one input vector through every node in the network.
        """
        if len(input) != self.input_size:
            raise ValueError("input must be the same size as the input size configuration")

        for i in range(len(self.nodes)):
            self.nodes[i].node_run(input)

        return [node.output for node in self.nodes]

    def backpropegate(self, new_weights, new_biases):
        """
        Passes weight and bias updates to each node in the network.
        """
        for i in range(len(self.nodes)):
            self.nodes[i].backpropegate(new_weights[i], new_biases[i])

class Lstm(Base):
    """
    Represents the top-level LSTM structure in the ML library.

    The class stores the shared input size and learning rate, creates a
    collection of nodes, runs input through each node, and forwards
    backpropagation updates down to those nodes.
    """
    
    def __init__(self, input_size, learning_rate, output_size=None):
        """
        Stores the LSTM configuration and creates its nodes.
        """
        super().__init__(input_size, learning_rate, output_size)

    def set_up(self):
        """
        Creates the node objects used by this LSTM.
        """
        for i in range(self.output_size):
            self.nodes.append(Node(i, self.input_size, self.learning_rate, LSTM=True))


class Linear(Base):
    """
    Represents a standard linear-style network built from non-LSTM nodes.
    """
    
    def __init__(self, input_size, learning_rate, bottlenck):
        self.bottlenck = bottlenck
        super().__init__(input_size, learning_rate, bottlenck)

    def set_up(self):
        """
        Creates the node objects used by this linear network.
        """
        for i in range(self.output_size):
            self.nodes.append(Node(i, self.input_size, self.learning_rate, LSTM=False))
