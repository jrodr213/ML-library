import math
import random    


# Node is the recurrent unit. It owns the memory value, output value, and
# four smaller Cell objects that act like gates.
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
        # Store basic node settings.
        self.node_number = node_number
        self.input_size = input_size
        self.learing_rate = learning_rate

        # These values are filled in when node_run() is called.
        self.input = None
        self.output = None
        self.cellstate = None

        # Each node creates its own forget, input, output, and candidate cells.
        self.cells = []
        self.create_node()
        

    def node_run(self, input):
        """
        Runs the node on an input vector and updates cellstate and output.
        """
        # Save the newest input so merge_input() can combine it with the
        # previous node output.
        self.input = input

        # The node only accepts fixed-size input vectors.
        if len(self.input) != self.input_size:
            raise ValueError("input must be the same size as the node input size")

        # The first time the node runs, it has no previous memory yet.
        if self.cellstate is None:
            self.cellstate = 0

        # The cells see both the previous node output and the current input.
        vector = self.merge_input()

        # Run the same merged input through all four cells.
        for i in range(len(self.cells)):
            # sets output for each cell, which is stored in the cell object
            self.cells[i].cell_run(vector)

        # Update the memory value:
        # old memory * forget gate + input gate * candidate value.
        self.cellstate = (
            (self.cellstate * self.cells[0].output)
            + (self.cells[1].output * self.cells[3].output)
        )

        # The output gate controls how much of the memory is visible as this
        # node's output.
        self.output = self.cells[2].tanh(self.cellstate) * self.cells[2].output

    def backpropegate(self, new_weights, new_bias):
        """
        Applies weight and bias updates to every cell in this node.
        """
        # Each cell receives the same update values and applies the learning
        # rate internally.
        for i in range(len(self.cells)):
            self.cells[i].update_weights_biases(self.learing_rate, new_weights, new_bias)       


    def create_node(self):
        """
        Creates the forget, input, output, and candidate cells for this node.
        """
        # Each cell receives one previous output value plus the current input
        # vector, so the merged input is input_size + 1.
        merged_input_size = self.input_size + 1

        # Cell 1 is the forget gate.
        cell1 = Cell(1, merged_input_size)
        self.cells.append(cell1)

        # Cell 2 is the input gate.
        cell2 = Cell(2, merged_input_size)
        self.cells.append(cell2)

        # Cell 3 is the output gate.
        cell3 = Cell(3, merged_input_size)
        self.cells.append(cell3)

        # Cell 4 is the candidate value cell.
        cell4 = Cell(4, merged_input_size)
        self.cells.append(cell4)


    def merge_input(self):
        """
        Combines the previous output value with the current input vector.
        """
        # On the first run there is no previous output, so use 0.
        if not self.output:
            previous_output = 0
        else:
            previous_output = self.output

        # Put previous output first, then the current input values.
        return [previous_output] + self.input
        

# Cell does one weighted calculation and then applies an activation function.
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
        # The form decides what type of cell this is.
        self.form = form

        # The input is stored each time cell_run() is called.
        self.input = None  
        self.input_size = input_size

        # Weights are created in cell_setup().
        self.weights = None

        # Forget cells start with a bias of 1 so they initially remember more.
        self.bias = 1 if self.form == 1 else 0

        # The cell output is created by cell_run().
        self.output = None
        self.cell_setup()

    def cell_run(self, input):
        """
        Runs the cell calculation on a list of numbers.

        The input is multiplied by the cell's weight vector, then summed
        with the bias. Forget, input, and output cells apply sigmoid to that
        value. Candidate cells apply tanh.
        """
        # Store this input so the cell can reference it later if needed.
        self.input = input

        # The input vector must match the number of weights.
        if len(self.input) != self.input_size:
            raise ValueError("input must be the same size as the cell weights")

        # Collapse the input vector into one weighted value.
        weighted_input = self.dot_product(self.input, self.weights) + self.bias
        
        # Candidate cells use tanh because their output can be negative.
        if self.form == 4:
            self.output = self.tanh(weighted_input)

        # Gate cells use sigmoid because their output should be between 0 and 1.
        else:
            self.output = self.sigmoid(weighted_input)

    def update_weights_biases(self, learning_rate, new_weights, new_bias):
        """
        Updates the cell weights based on the error and learning rate.
        """
        # Move each weight in the opposite direction of its update value.
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - (learning_rate * new_weights[i])

        # Update the bias the same way as the weights.
        self.bias = self.bias - (learning_rate * new_bias)


    def cell_setup(self):
        """
        Validates the cell form and creates the starting weights.
        """
        # Only the four known cell types are allowed.
        if self.form < 1 or self.form > 4:
            raise ValueError("form must be between 1 and 4")
        
        # Form numbers must be integers so comparisons and cell selection work.
        if not isinstance(self.form, int):
            raise TypeError("form must be an integer")
        
        # Start weights near zero so no cell begins too strongly.
        self.weights = [
            random.uniform(-0.01, 0.01)
            for _ in range(self.input_size)
        ]

    def dot_product(self, input, weights):
        """
        Multiplies each input by its matching weight and sums the results.
        """
        # zip() pairs each input value with its matching weight.
        return sum(input_value * weight for input_value, weight in zip(input, weights))
            

    def sigmoid(self, value):
        """
        Applies the sigmoid function to one number.
        """
        # Sigmoid maps any number into the range 0 to 1.
        return 1 / (1 + (math.e ** (value * -1)))
    
    def tanh(self, value):
        """
        Applies the tanh function to one number.
        """
        # tanh maps any number into the range -1 to 1.
        top = (math.e ** value) - (math.e ** (-1 * value))
        bottom = (math.e ** value) + (math.e ** (-1 * value))
        return top / bottom

    
