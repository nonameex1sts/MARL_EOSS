import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from prj_utils.utils import evaluate_assignment
import numpy as np
import gym

# This environment simulates a task scheduling problem where multiple agents must select tasks to maximize rewards
class CustomEnv(gym.Env):
    def __init__(self, file_path: str, target_assignments: np.ndarray):
        super(CustomEnv, self).__init__()

        self.target_assignments = target_assignments
        self.target_id = np.random.randint(0, len(self.target_assignments)) if self.target_assignments is not None else 0 # Randomly select a target assignment

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

        # Initialize state and rewards
        self.state = None
        self.loss = None

    def reset(self):
        # Reset the environment state and rewards
        self.state = np.negative(np.ones(self.num_tasks, dtype=int))  # -1 indicates unassigned tasks
        self.loss = np.zeros(self.num_agents, dtype=float)
        self.target_id = np.random.randint(0, len(self.target_assignments)) if self.target_assignments is not None else 0  # Randomly select a target assignment
        # self.target_id = 0
        return self.state
    
    def update(self, new_task: np.ndarray):
        # Update the number of tasks
        self.num_tasks += 1
        self.rewards = np.insert(self.rewards, new_task['id'], [new_task['reward']])  # Insert the new task's priority
        
        # Update the conflicts matrix
        new_conflict = np.zeros(self.num_tasks, dtype=int)
        for conflict in new_task['conflict_set']:
            new_conflict[conflict] = 1

        self.conflicts_matrix = np.insert(self.conflicts_matrix, new_task['id'], np.delete(new_conflict, new_task['id']), axis=0) # Insert the new row for the new task
        self.conflicts_matrix = np.insert(self.conflicts_matrix, new_task['id'], new_conflict, axis=1) # Insert the new column for the new task
    
    def check_valid_state(self) -> bool:
        # Return True if the current state has no conflicts with other tasks or if the task is unassigned
        for i in range(1, self.num_agents + 1):
            # Identify the tasks assigned to agent i
            agent_assignment = self.state == i

            # Check for conflicts between tasks assigned to the same agent
            for j in range(self.num_tasks):
                for k in range(j + 1, self.num_tasks):
                    if agent_assignment[j] and agent_assignment[k] and self.conflicts_matrix[j][k] == 1:
                        return False
        return True

    def step(self, task_index, chosen_agent):
        self.state[task_index] = chosen_agent  # Assign the task to the chosen agent
        self.loss = np.zeros(self.num_agents, dtype=float)

        # Compare the chosen agent with the target assignment
        if self.target_assignments is not None:
            self.state[task_index] = self.target_assignments[self.target_id][task_index]

            self.loss[self.target_assignments[self.target_id][task_index]] = 1
            if self.loss[chosen_agent] == 0:
                # If the agent did not receive a reward, penalize it
                self.loss[chosen_agent] = -1
          
        return self.state, self.loss, task_index >= self.num_tasks - 1
            
    def get_task_info(self, task_index) -> tuple[float, np.ndarray]:
        # Get the reward and conflicts of the task
        reward = self.rewards[task_index]
        conflicts = self.conflicts_matrix[task_index]
        return reward, conflicts
    
    def evaluate(self) -> np.ndarray:
        """
        Evaluate the assignment
        """
        return evaluate_assignment(num_tasks=self.num_tasks, rewards=self.rewards, assignment=self.state)