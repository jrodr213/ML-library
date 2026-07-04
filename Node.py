import math
import random    


class Node:
    """
    Represents one recurrent node in the ML library.

    A node stores the latest output value and cell state value. Each run
    sends the merged input through either four LSTM cells or one standard
    cell:
        1: forget cell, which controls how much old cell state remains
        2: input cell, which gates new candidate state values
        3: output cell, which gates the visible node output
        4: candidate cell, which creates new candidate state values
        5: standard cell, which returns the raw dot product

    Attributes:
        node_number: Identifier for this node.
        input_size: The fixed size of vectors this node accepts.
        output: Most recent output value produced by the node.
        cellstate: Internal memory value carried between runs.
        cells: The cells used by the node.
    """
    def __init__(self, node_number, input_size, learning_rate, LSTM=False):
        self.node_number = node_number
        self.input_size = input_size
        self.learing_rate = learning_rate
        self.LSTM = LSTM
        self.output = None
        self.cellstate = None
        self.cells = []
        self.create_node()

    def node_run(self, input, previous_hidden=None):
        """
        Runs the node on an input vector and updates cellstate and output.
        """
        if len(input) != self.input_size:
            raise ValueError("input must be the same size as the node input size")

        if previous_hidden is None:
            previous_hidden = [0]
        elif not isinstance(previous_hidden, list):
            previous_hidden = [previous_hidden]

        if len(previous_hidden) != 1:
            raise ValueError("previous_hidden must contain exactly one value")

        if self.cellstate is None:
            self.cellstate = 0

        vector = previous_hidden + input

        for i in range(len(self.cells)):
            # sets output for each cell, which is stored in the cell object
            self.cells[i].cell_run(vector)

        if not self.LSTM:
            self.output = self.cells[0].output
            

        else:    
            self.cellstate = (
                (self.cellstate * self.cells[0].output)
                + (self.cells[1].output * self.cells[3].output)
            )

            self.output = self.cells[2].tanh(self.cellstate) * self.cells[2].output

    def backpropegate(self, new_weights, new_bias):
        """
        Applies weight and bias updates to every cell in this node.
        """
        for i in range(len(self.cells)):
            self.cells[i].update_weights_biases(self.learing_rate, new_weights, new_bias)       


    def create_node(self):
        """
        Creates the cells used by this node.
        """
        merged_input_size = self.input_size + 1

        if not self.LSTM:
            cell5 = Cell(5, merged_input_size)
            self.cells.append(cell5)
            
        else:
            cell1 = Cell(1, merged_input_size)
            self.cells.append(cell1)

            cell2 = Cell(2, merged_input_size)
            self.cells.append(cell2)

            cell3 = Cell(3, merged_input_size)
            self.cells.append(cell3)

            cell4 = Cell(4, merged_input_size)
            self.cells.append(cell4)


class Cell:
    """
    Represents one calculation cell in the ML library.

    A cell is created with a form number and fixed input size, then receives
    a matching list of input values when cell_run() is called. The form
    controls how the list is processed, and the result is stored in
    self.output.

    Attributes:
        form: The type of cell to create.
            1: forget cell
            2: input cell
            3: output cell
            4: candidate cell
            5: standard cell
        weights: The cell weights. Each weight starts as a small random value.
        input_size: The fixed size of the input vector after setup.
        bias: The cell bias. Forget cells start at 1, and other cells
            start at 0.
        output: The most recent output from cell_run().
    """

    def __init__(self, form, input_size):      
        self.form = form
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
        value. Candidate cells apply tanh. Standard cells return the raw
        dot product plus bias.
        """
        weighted_input = self.dot_product(input, self.weights) + self.bias
        
        if self.form == 4:
            self.output = self.tanh(weighted_input)

        elif self.form == 5:
            self.output = weighted_input

        else:
            self.output = self.sigmoid(weighted_input)

    def update_weights_biases(self, learning_rate, new_weights, new_bias):
        """
        Updates the cell weights based on the error and learning rate.
        """
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - (learning_rate * new_weights[i])

        self.bias = self.bias - (learning_rate * new_bias)


    def cell_setup(self):
        """
        Validates the cell form and creates the starting weights.
        """
        if self.form < 1 or self.form > 5:
            raise ValueError("form must be between 1 and 5")
        
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

    
