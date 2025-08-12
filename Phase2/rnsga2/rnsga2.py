import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from Phase1.nsga2.problem import EOSSProblem
from pymoo.optimize import minimize
import numpy as np

# Function to create a population of identical individuals based on an initial assignment
class IdenticalSampling(IntegerRandomSampling):
    def __init__(self, initial_assignment=None):
        super().__init__()
        self.initial_assignment = np.concatenate((np.array([0]), initial_assignment))

    def _do(self, problem, n_samples, **kwargs):
        return np.array([self.initial_assignment for _ in range(n_samples)])
    
class RNSGA2_Solver:
    def __init__(self, file_path, **kwargs):
        self.problem = EOSSProblem(file_path)

    def solve(self, current_assignment, new_task, ref_point, pop_size=100, n_gen=500, seed=None, verbose=True, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        # Update the problem with the new task and current assignment
        self.problem._update(new_task, current_assignment)

        # Initialize the NSGA-II algorithm for discrete optimization
        algorithm = RNSGA2(ref_points=-np.array([ref_point]),
                        pop_size=pop_size, 
                        sampling=IdenticalSampling(current_assignment[new_task['id']:]),
                        crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                        mutation=PM(prob=0.5, eta=3.0, vtype=float, repair=RoundingRepair()),
                        eliminate_duplicates=True,
                        epsilon=0.01,
                        normalization='front',
                        extreme_points_as_reference_points=False,
                        weights=np.array([0.5, 0.5]))

        # Run the optimization
        res = minimize(self.problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=verbose)

        return res.X[0], -res.F[0]  # Return the first solution and its objectives