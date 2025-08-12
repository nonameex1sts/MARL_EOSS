import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from prj_utils.utils import evaluate_assignment
import numpy as np

class Greedy_Solver:
    def __init__(self, file_path, **kwargs):
        # Read the dataset from the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines

            # First line contains the number of tasks and agents
            self.num_agents, self.num_tasks = map(int, lines[0].split())
            self.num_agents += 1  # Include the no-assignment decision

            # Second line contains the rewards of the tasks
            self.rewards = np.array(list(map(int, lines[1].split())))

            # Third line onwards contains the conflicts between tasks:
            # Line i contains the indices of tasks that conflict with task i-2 or -1 if no conflicts
            self.conflicts_matrix = np.zeros((self.num_tasks, self.num_tasks), dtype=int)
            for i in range(2, 2 + self.num_tasks):
                confict = np.array(list(map(int, lines[i].split())))
                if confict[0] != -1:
                    for j in confict:
                        self.conflicts_matrix[i - 2][j] = 1
                        self.conflicts_matrix[j][i - 2] = 1

    def update(self, new_task):
        """
        Update the problem with the new task.
        """
        # Update the number of tasks
        self.num_tasks += 1
        self.rewards = np.insert(self.rewards, new_task['id'], [new_task['reward']])  # Insert the new task's priority
        
        # Update the conflicts matrix
        new_conflict = np.zeros(self.num_tasks, dtype=int)
        for conflict in new_task['conflict_set']:
            new_conflict[conflict] = 1

        self.conflicts_matrix = np.insert(self.conflicts_matrix, new_task['id'], np.delete(new_conflict, new_task['id']), axis=0) # Insert the new row for the new task
        self.conflicts_matrix = np.insert(self.conflicts_matrix, new_task['id'], new_conflict, axis=1) # Insert the new column for the new task

    def get_feasible_agents(self, task, assignment):
        """
        Get the feasible agents for a given task based on the current assignment and conflicts.
        """
        feasible_agents = set(range(1, self.num_agents + 1))  # All agents are initially feasible
        # Remove agents assigned to conflicting tasks
        for task_id in range(self.num_tasks):
            if self.conflicts_matrix[task][task_id] == 1:
                feasible_agents.discard(assignment[task_id])
        return list(feasible_agents)
    
    def solve(self, current_assignment, new_task, ref_point, verbose=True, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the problem with the new task and current assignment.
        """
        # Update the number of tasks
        self.update(new_task)

        # Initialize the assignment with the current assignment
        new_assignment = np.zeros(self.num_tasks, dtype=int)
        new_assignment[:new_task['id']] = current_assignment[:new_task['id']]  # Keep the current assignment for completed tasks

        # Greedily assign the remaining tasks to agents
        for task in range(new_task['id'], self.num_tasks):
            feasible_agents = self.get_feasible_agents(task, new_assignment)
            if len(feasible_agents) > 0:
                # Select a random agent from the feasible agents
                new_assignment[task] = np.random.choice(feasible_agents)

        return new_assignment[new_task['id']:], evaluate_assignment(num_tasks=self.num_tasks, rewards=self.rewards, assignment=new_assignment)