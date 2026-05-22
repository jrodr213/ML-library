import math
import random
from numbers import Number


class Node:
    """
    This class represents the structure of a node in this
    ML library.

    Attributes:
    """
    def __init__(self, node_number):
        self.node_number = node_number
        self.input = None
        self.output = None
        self.cellstate = None
        self.cells = []
        self.create_node()

    def node_run(self, input):
        self.input = input
        vector = self.merge_input()
        for i in range(len(self.cells)):
            self.cells[i].cell_run(self.input)


    def create_node(self):
        cell1 = Cell(1)
        self.cells.append(cell1)
        cell2 = Cell(2)
        self.cells.append(cell2)
        cell3 = Cell(3)
        self.cells.append(cell3)


    def merge_input(self):
        if not self.output:
            return self.input
        else:
            for i in range(self.input):
                self.input[i] = self.input[i] + self.output[i]
            return self.input
        

class Cell:
    """
    Represents one calculation cell in the ML library.

    A cell is created with a form number, then receives a list of input
    values when cell_run() is called. The form controls how the list is
    processed, and the result is stored in self.output.

    Attributes:
        input: The most recent list of numbers passed into cell_run().
        form: The type of cell to create.
            1: forget cell
            2: input cell
            3: output cell
        weight: The cell weight. Forget cells use 1, and other cells
            start with a small random value.
        bias: The cell bias. This currently starts at 0.
        output: The most recent output from cell_run().
    """

    def __init__(self, form):      
        self.form = form
        self.input = None  
        self.weight = None
        self.bias = 0
        self.output = None
        self.cell_setup()

    def cell_run(self, input):
        """
        Runs the cell calculation on a list of numbers.

        Forget and output cells apply sigmoid to each value. Input cells
        multiply each sigmoid result by the matching tanh result.
        """
        self.input = input
        if self.form == 1 or self.form == 3:
            self.output = self.sigmoid(self.input)
        
        if self.form == 2:
            self.output = [
                sigmoid_value * tanh_value
                for sigmoid_value, tanh_value in zip(
                    self.sigmoid(self.input),
                    self.tanh(self.input)
                )
            ]
        

    def cell_setup(self):
        """
        Validates the cell form and creates the starting weight.
        """
        if self.form < 1 or self.form > 3:
            raise ValueError("form must be between 1 and 3")
        
        if not isinstance(self.form, int):
            raise TypeError("form must be an integer")
        
        if self.form == 1:
            self.weight = 1

        else:
            self.weight = random.uniform(-0.01, 0.01)
            

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

    
