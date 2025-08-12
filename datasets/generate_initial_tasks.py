import numpy as np

# Parameters for dataset generation
n = 5 # Number of agents
m = 500 # Number of tasks
levels = 3 # Number of priority levels
priority_probabilities = [0.4, 0.4, 0.2]  # Probabilities for each priority level
confict_lengths = [0, 9, 19] # Conflict lengths for priorities 1, 2, and 3
file_name = f'datasets/n{n}_m{m}.txt'
map_priorities = [1, 10, 30]  # Mapping priorities to values (1, 10, 30)

"""
Confic rules:
1. Low priority tasks (priority 1) have no conflicts.
2. Medium and high priority tasks (priority 2) conflict with other tasks with the same or lower priority in the conflict length range.
3. Conflict probability (for medium and high priority): closer tasks to the current task have higher chance of conflict
"""
with open(file_name, 'w') as f:
    f.write(f"{n} {m}\n")  # Write the first line with n, m

    priorities = [np.random.choice(levels, p=priority_probabilities) for _ in range(m)]  # Random priorities between 1 and levels
    f.write(" ".join(map(str, [map_priorities[p] for p in priorities])) + "\n")  # Write the mapped priorities

    for i, priority in enumerate(priorities):
        # Generate conflicts for each task based on its priority
        if priority == 0:
            # Low priority tasks have no conflicts
            f.write("-1\n")
        else:
            confict_length = confict_lengths[priority]
            conflict_task = []
            for j in range(max(0, i - int(confict_length / 2)), min(m, i + int(confict_length / 2) + 1)):
                if i != j:
                    conflict_task.append(j)
            
            # Reverse the conflict task list to maintain order
            # conflict_task.reverse()
            if not conflict_task:
                f.write("-1\n")
            else:
                f.write(" ".join(map(str, conflict_task)) + "\n")