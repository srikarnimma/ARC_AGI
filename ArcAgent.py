import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet


class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init method. Be aware that it gets called only once
        and then the make_predictions method will get called several times.
        """
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        """
        Write the code in this method to solve the incoming ArcProblem.
        Your agent will receive 1 problem at a time.

        You can add up to THREE (3) the predictions to the
        predictions list provided below that you need to
        return at the end of this method.

        In the Autograder, the test data output in the arc problem will be set to None
        so your agent cannot peek at the answer (even on the public problems).

        Also, if you return more than 3 predictions in the list it
        is considered an ERROR and the test will be automatically
        marked as INCORRECT.
        """

        # Initialize input, output, and predictions list
        predictions: list[np.ndarray] = list()
        training_data = arc_problem.training_set()
        training_input = [data.get_input_data().data() for data in training_data]
        training_output = [data.get_output_data().data() for data in training_data] 

        test_data = arc_problem.test_set()
        test_input = test_data.get_input_data().data()

        # Hardcoded solution to Milestone B question 1:
        nonzero_positions = np.where(test_input != 0)
        
        if len(nonzero_positions[0]) > 0:
            min_row = np.min(nonzero_positions[0])
            max_row = np.max(nonzero_positions[0])
            min_col = np.min(nonzero_positions[1])
            max_col = np.max(nonzero_positions[1])
            
            # Extract the smallest submatrix containing all nonzero entries
            result = test_input[min_row:max_row+1, min_col:max_col+1]
            predictions.append(result)
        else:
            predictions.append(np.array([[]]))

        return predictions
