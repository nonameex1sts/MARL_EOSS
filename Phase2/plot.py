import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 30})

num_agents = 3
num_tasks = 50
folder_name = "Small scale" if num_tasks < 100 else "Large scale"
runtimes = [[] for _ in range(3)]  # Store runtimes for each algorithm
similarities = [[] for _ in range(3)]  # Store similarities for each algorithm
for i, alg in enumerate(["RNSGA-II", "Greedy", "QMIX"]):
    with open(f"Phase2/results/{folder_name}/n{num_agents}_m{num_tasks}/{alg}.txt", 'r') as f:
        for line in f:
            parts = line.strip().split()
            runtimes[i].append(float(parts[0]))
            similarities[i].append(float(parts[1]))

# Print the results in tabular format
print(f"{'Algorithm':<10} {'Mean Runtime (log10 seconds)':<40} {'Mean Similarity':<20}")
for i, alg in enumerate(["RNSGA-II", "Greedy", "QMIX"]):
    mean_runtime = np.mean(runtimes[i])
    mean_similarity = np.mean(similarities[i])
    print(f"{alg:<10} {mean_runtime:<40.4f} {mean_similarity:<20.4f}")


# Plotting the Similarities as box plots
plt.figure(figsize=(12, 6))

box = plt.boxplot(similarities, labels=["RNSGA-II", "Greedy", "QMIX"])
for element in ['boxes', 'whiskers', 'caps', 'medians']:
    for line in box[element]:
        line.set_linewidth(1.5)  # Adjust thickness here
# plt.title("Similarities to Initial Schedule")
plt.ylabel("Similarities to Initial Schedule")
plt.tight_layout()
plt.show()