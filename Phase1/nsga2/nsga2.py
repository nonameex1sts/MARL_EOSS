from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from .problem import EOSSProblem
from pymoo.optimize import minimize
import numpy as np

# Function to create a population of all-zero individuals
class ZeroSampling(IntegerRandomSampling):

    def _do(self, problem, n_samples, **kwargs):
        n = problem.n_var
        return np.column_stack([np.zeros(n_samples) for _ in range(n)])
    
def NSGA2_solve(file_path, pop_size=300, n_gen=1000, seed=None, verbose=True):
    # Create an instance of the problem
    problem = EOSSProblem(file_path)

    # Initialize the NSGA-II algorithm for discrete optimization
    algorithm = NSGA2(pop_size=pop_size, 
                      sampling=ZeroSampling(),
                      crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                      mutation=PM(prob=0.5, eta=3.0, vtype=float, repair=RoundingRepair()),
                      eliminate_duplicates=True)

    # Run the optimization
    res = minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=verbose)

    return res