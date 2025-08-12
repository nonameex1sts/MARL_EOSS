import numpy as np

# Parameters for dataset generation
n = 6 # Number of agents
m = 500 # Number of tasks
inc = int(0.2 * m)  # Number of incoming tasks
levels = 3 # Number of priority levels
priority_probabilities = [0.2, 0.4, 0.4]  # Probabilities for each priority level
confict_lengths = [0, 9, 19] # Conflict lengths for priorities 1, 2, and 3
file_name = f'datasets/n{n}_m{m}_inc{inc}.txt'
map_priorities = [1, 10, 30]  # Mapping priorities to values (1, 10, 30)

with open(file_name, 'w') as f:
    f.write(f"{inc}\n")  # Write the first line with the number of incoming tasks

    positions = np.random.choice(m, size=int(inc), replace=False)  # Randomly select positions for incoming tasks
    positions.sort()  # Sort the positions to maintain order
    f.write(" ".join(map(str, positions)) + "\n")  # Write the positions of incoming tasks

    priorities = [np.random.choice(levels, p=priority_probabilities) for _ in range(inc)]  # Random priorities between 1 and levels
    f.write(" ".join(map(str, [map_priorities[p] for p in priorities])) + "\n")  # Write the mapped priorities

    for i, priority in enumerate(priorities):
        # Generate conflicts for each task based on its priority
        if priority == 0:
            # Low priority tasks have no conflicts
            f.write("-1\n")
        else:
            confict_length = confict_lengths[priority]
            conflict_task = []
            for j in range(max(0, positions[i] - int(confict_length / 2)), min(m + i, positions[i] + int(confict_length / 2) + 1)):
                if positions[i] != j:
                    conflict_task.append(j)
            
            # Reverse the conflict task list to maintain order
            # conflict_task.reverse()
            if not conflict_task:
                f.write("-1\n")
            else:
                f.write(" ".join(map(str, conflict_task)) + "\n")