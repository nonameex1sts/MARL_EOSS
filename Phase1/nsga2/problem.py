import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from prj_utils.utils import evaluate_assignment
from pymoo.core.problem import Problem
import numpy as np

# Problem class for the EOSS problem, which inherits from pymoo's Problem class to use in NSGA-II and RNSGA-II algorithms
class EOSSProblem(Problem):
    def __init__(self, file_path: str):
        # Read the dataset from the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines

            # First line contains the number of tasks and agents
            self.num_agents, self.num_tasks = map(int, lines[0].split())

            # Second line contains the rewards of the tasks
            self.rewards = np.array(list(map(int, lines[1].split())))

            # In each set, the first number is the task ID, followed by the IDs of all tasks conflicting with it
            self.conflicts_sets = []
            for i in range(2, 2 + self.num_tasks):
                confict = list(map(int, lines[i].split()))
                if confict[0] != -1:
                    confict.insert(0, i - 2)  # Insert the main task ID at the beginning
                    self.conflicts_sets.append(np.array(confict))
        
        # Initialize the past assignment as an empty array (for phase 2)
        self.past_assignment = np.array([])

        #Initailize the problem
        super().__init__(n_var=self.num_tasks, n_obj=2, n_ieq_constr=len(self.conflicts_sets), 
                         xl=np.zeros(self.num_tasks), xu=np.array([self.num_agents for _ in range(self.num_tasks)]), vtype=int)
        
    def _update(self, new_task: map, current_assignment: np.ndarray):
        # Update the rewards with a new task
        self.rewards = np.insert(self.rewards, new_task['id'], [new_task['reward']])  # Insert the new task's priority
        
        # Update the conflict sets according to the new task (push all tasks after the new task 1 position forward)
        for i in range(len(self.conflicts_sets)):
            for j in range(len(self.conflicts_sets[i])):
                if self.conflicts_sets[i][j] >= new_task['id']:
                    self.conflicts_sets[i][j] += 1

        # Add the new task's conflict set
        if len(new_task['conflict_set']) > 0:
            self.conflicts_sets.append(np.array([new_task['id']] + new_task['conflict_set']))
            self.n_ieq_constr += 1  # Increase the number of inequality constraints

        self.past_assignment = np.array(current_assignment[:new_task['id']])  # Store the past assignment up to the new task's ID

        # Update the problem's variables
        self.num_tasks += 1  # Increase the number of tasks
        self.n_var = self.num_tasks - new_task['id']  # Update the number of variables
        self.xl = np.zeros(self.n_var)  # Reset the lower bound
        self.xu = np.array([self.num_agents for _ in range(self.n_var)])
        self.new_id = new_task['id']  # Store the ID of the new task

    def _evaluate(self, x, out, *args, **kwargs):
        # Calculate penalties for conflicts and objectives (completion rate and reward percentage)
        completion_rate = []
        reward_percentages = []
        penalties = []
        for ans in x:
            metric = evaluate_assignment(num_tasks=self.num_tasks, rewards=self.rewards, assignment=np.concatenate((self.past_assignment, ans)))
            completion_rate.append(metric[0])  # Completion rate
            reward_percentages.append(metric[1]) # Reward percentage

            penalty = np.zeros(self.n_ieq_constr, dtype=int) # Initialize penalties for each conflict set

            assignment = np.concatenate((self.past_assignment, ans)) # Construct the full assignment including past tasks and the updated tasks

            # Iterate through each conflict set
            for i in range(self.n_ieq_constr):
                task_id = self.conflicts_sets[i][0] # Find the main task in the conflict set
                # Find all tasks conflicting with the main task
                if assignment[task_id] != 0:
                    for id in self.conflicts_sets[i][1:]:
                        if assignment[id] == assignment[task_id]:
                            penalty[i] += 1 # Count the number of conflicts with the main task
            
            penalties.append(penalty)

        # Set the objective values
        out["F"] = np.column_stack((-np.array(completion_rate), -np.array(reward_percentages)))
        out["G"] = np.array(penalties)