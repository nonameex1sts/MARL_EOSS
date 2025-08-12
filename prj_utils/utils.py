import numpy as np

def evaluate_assignment(num_tasks, rewards, assignment) -> np.ndarray:
    """
    Evaluate the assignment
    """
    # Calculate the completion rate
    completion_rate = np.sum(assignment != 0) / num_tasks

    # Calculate the percentage of rewards achieved
    reward_percentage = np.sum((assignment != 0) * rewards) / np.sum(rewards)
    
    return np.array([completion_rate, reward_percentage])