import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Phase1.qmix.qmix import QMIX
import numpy as np

class QMIX_Solver:
    def __init__(self, file_path, model_path, **kwargs):
        self.qmix_solver = QMIX(file_path)
        self.qmix_solver.load_model(model_path)  # Load the pre-trained model

    def solve(self, current_assignment, new_task, ref_point=None, verbose=True, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the problem using QMix.
        """
        return self.qmix_solver.plan(current_assignment, new_task, ref_point=ref_point, verbose=verbose)