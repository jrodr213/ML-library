import math
import random
from numbers import Number

class Model:
    """
    Represents the ML model in the library.

    The model is a collection of nodes, which are created when the model is
    initialized. The model can be run on an input vector, which sends the
    vector through each node and produces an output vector.

    Attributes:
        input_size: The fixed size of input vectors passed into the model.
        nodes: A list of Node objects that make up the model.
    """
    def __init__(self, num_nodes, input_size):
        self.input_size = input_size
        self.nodes = []
        self.create_model(num_nodes)

    def create_model(self, num_nodes):
        """
        Creates the specified number of nodes for the model.
        """
        for i in range(num_nodes):
            node = Node(i, self.input_size)
            self.nodes.append(node)

    def model_run(self, input):
        """
        Runs the model on an input vector and returns the output vector.
        """
        for node in self.nodes:
            node.node_run(input)
            output = node.output
      
        

class Node:
    """
    Represents one recurrent node in the ML library.

    A node stores the latest input vector, output vector, and cell state
    vector. Each run sends the merged input through three cells:
        1: forget cell, which controls how much old cell state remains
        2: input cell, which creates new candidate state values
        3: output cell, which gates the visible node output

    Attributes:
        node_number: Identifier for this node.
        input_size: The fixed size of vectors this node accepts.
        input: Most recent input vector passed to node_run().
        output: Most recent output vector produced by the node.
        cellstate: Internal memory vector carried between runs.
        cells: The forget, input, and output cells used by the node.
    """
    def __init__(self, node_number, input_size):
        self.node_number = node_number
        self.input_size = input_size
        self.input = None
        self.output = None
        self.cellstate = None
        self.cells = []
        self.create_node()

    def node_run(self, input):
        """
        Runs the node on an input vector and updates cellstate and output.
        """
        self.input = input
        if len(self.input) != self.input_size:
            raise ValueError("input must be the same size as the node input size")

        if self.cellstate is None:
            self.cellstate = [0 for _ in self.input]
        vector = self.merge_input()

        for i in range(len(self.cells)):
            # sets output for each cell, which is stored in the cell object
            self.cells[i].cell_run(vector)

        self.cellstate = [
            (state_value * self.cells[0].output) + self.cells[1].output
            for state_value in self.cellstate
        ]

        self.output = [
            tanh_value * self.cells[2].output
            for tanh_value in self.cells[2].tanh(self.cellstate)
        ]


    def create_node(self):
        """
        Creates the forget, input, and output cells for this node.
        """
        cell1 = Cell(1, self.input_size)
        self.cells.append(cell1)
        cell2 = Cell(2, self.input_size)
        self.cells.append(cell2)
        cell3 = Cell(3, self.input_size)
        self.cells.append(cell3)


    def merge_input(self):
        """
        Combines the current input vector with the previous output vector.
        """
        if not self.output:
            return self.input
        else:
            for i in range(len(self.input)):
                self.input[i] = self.input[i] + self.output[i]
            return self.input
        

class Cell:
    """
    Represents one calculation cell in the ML library.

    A cell is created with a form number and fixed input size, then receives
    a matching list of input values when cell_run() is called. The form
    controls how the list is processed, and the result is stored in
    self.output.

    Attributes:
        input: The most recent list of numbers passed into cell_run().
        form: The type of cell to create.
            1: forget cell
            2: input cell
            3: output cell
        weights: The cell weights. Each weight starts as a small random value.
        input_size: The fixed size of the input vector after setup.
        bias: The cell bias. Forget cells start at 1, and other cells
            start at 0.
        output: The most recent output from cell_run().
    """

    def __init__(self, form, input_size):      
        self.form = form
        self.input = None  
        self.input_size = input_size
        self.weights = None
        self.bias = 1 if self.form == 1 else 0
        self.output = None
        self.cell_setup()

    def cell_run(self, input):
        """
        Runs the cell calculation on a list of numbers.

        The input is multiplied by the cell's weight vector, then summed
        with the bias. Forget and output cells apply sigmoid to that value.
        Input cells multiply sigmoid(value) by tanh(value).
        """
        self.input = input
        if len(self.input) != self.input_size:
            raise ValueError("input must be the same size as the cell weights")

        weighted_input = self.dot_product(self.input, self.weights) + self.bias

        if self.form == 1 or self.form == 3:
            self.output = self.sigmoid([weighted_input])[0]
        
        if self.form == 2:
            self.output = self.sigmoid([weighted_input])[0] * self.tanh([weighted_input])[0]
        

    def cell_setup(self):
        """
        Validates the cell form and creates the starting weights.
        """
        if self.form < 1 or self.form > 3:
            raise ValueError("form must be between 1 and 3")
        
        if not isinstance(self.form, int):
            raise TypeError("form must be an integer")
        
        self.weights = [
            random.uniform(-0.01, 0.01)
            for _ in range(self.input_size)
        ]

    def dot_product(self, input, weights):
        """
        Multiplies each input by its matching weight and sums the results.
        """
        return sum(input_value * weight for input_value, weight in zip(input, weights))
            

    def sigmoid(self, input):
        """
        Applies the sigmoid function to each number in the input list.
        """
        return [(1/(1+ (math.e ** (value * -1)))) for value in input]
    
    def tanh(self, input):
        """
        Applies the tanh function to each number in the input list.
        """
        output = []
        for value in input:
            top = (math.e ** value) - (math.e ** (-1 * value))
            bottom = (math.e ** value) + (math.e ** (-1 * value))
            output.append(top/bottom)
        return output

    
