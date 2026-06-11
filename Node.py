import math
import random    

class Node:
    """
    Represents one recurrent node in the ML library.

    A node stores the latest input vector, output value, and cell state
    value. Each run sends the merged input through four cells:
        1: forget cell, which controls how much old cell state remains
        2: input cell, which gates new candidate state values
        3: output cell, which gates the visible node output
        4: candidate cell, which creates new candidate state values

    Attributes:
        node_number: Identifier for this node.
        input_size: The fixed size of vectors this node accepts.
        input: Most recent input vector passed to node_run().
        output: Most recent output value produced by the node.
        cellstate: Internal memory value carried between runs.
        cells: The forget, input, output, and candidate cells used by the node.
    """
    def __init__(self, node_number, input_size, learning_rate):
        self.node_number = node_number
        self.input_size = input_size
        self.learing_rate = learning_rate
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
            self.cellstate = 0
        vector = self.merge_input()

        for i in range(len(self.cells)):
            # sets output for each cell, which is stored in the cell object
            self.cells[i].cell_run(vector)

        self.cellstate = (
            (self.cellstate * self.cells[0].output)
            + (self.cells[1].output * self.cells[3].output)
        )

        self.output = self.cells[2].tanh(self.cellstate) * self.cells[2].output

    def backpropegate(self):
        """
        Placeholder for backpropegation function.
        """
        pass           


    def create_node(self):
        """
        Creates the forget, input, output, and candidate cells for this node.
        """
        merged_input_size = self.input_size + 1
        cell1 = Cell(1, merged_input_size)
        self.cells.append(cell1)
        cell2 = Cell(2, merged_input_size)
        self.cells.append(cell2)
        cell3 = Cell(3, merged_input_size)
        self.cells.append(cell3)
        cell4 = Cell(4, merged_input_size)
        self.cells.append(cell4)


    def merge_input(self):
        """
        Combines the previous output value with the current input vector.
        """
        if not self.output:
            previous_output = 0
        else:
            previous_output = self.output

        return [previous_output] + self.input
        

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
            4: candidate cell
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
        with the bias. Forget, input, and output cells apply sigmoid to that
        value. Candidate cells apply tanh.
        """
        self.input = input
        if len(self.input) != self.input_size:
            raise ValueError("input must be the same size as the cell weights")

        weighted_input = self.dot_product(self.input, self.weights) + self.bias
        
        if self.form == 4:
            self.output = self.tanh(weighted_input)

        else:
            self.output = self.sigmoid(weighted_input)
        

    def cell_setup(self):
        """
        Validates the cell form and creates the starting weights.
        """
        if self.form < 1 or self.form > 4:
            raise ValueError("form must be between 1 and 4")
        
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
            

    def sigmoid(self, value):
        """
        Applies the sigmoid function to one number.
        """
        return 1 / (1 + (math.e ** (value * -1)))
    
    def tanh(self, value):
        """
        Applies the tanh function to one number.
        """
        top = (math.e ** value) - (math.e ** (-1 * value))
        bottom = (math.e ** value) + (math.e ** (-1 * value))
        return top / bottom

    
