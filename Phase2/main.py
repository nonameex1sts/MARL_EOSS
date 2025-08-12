import numpy as np
import os
import time
from rnsga2.rnsga2 import RNSGA2_Solver
from greedy.greedy import Greedy_Solver
from qmix.qmix import QMIX_Solver

if __name__ == "__main__":
    # Define the size of the dataset
    num_agents = 4
    num_tasks = 100
    incoming_tasks = int(0.2 * num_tasks)  # Number of additional tasks to simulate

    # Determine the folder name based on the number of tasks
    folder_name = "Small scale" if num_tasks < 100 else "Large scale"
    initial_dataset = f"datasets/{folder_name}/n{num_agents}_m{num_tasks}.txt"
    additional_dataset = f"datasets/{folder_name}/n{num_agents}_m{num_tasks}_inc{incoming_tasks}.txt"

    # Create the results folder if it does not exist
    results_folder = f"Phase2/results/{folder_name}/n{num_agents}_m{num_tasks}/"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    algorithms = {
        "RNSGA-II": RNSGA2_Solver,
        "Greedy": Greedy_Solver,
        "QMIX": QMIX_Solver
    }

    for algo_name, Solver in algorithms.items():
        for i in range(30):
            phase1_folder = f"Phase1/results/{folder_name}/n{num_agents}_m{num_tasks}/{i+1}"

            # Load the chosen assignment and its objectives from the saved files
            chosen_assignment = np.load(f"{phase1_folder}/chosen_assignment.npy")
            chosen_objectives = np.load(f"{phase1_folder}/chosen_objectives.npy")

            # Print the chosen assignment and its objectives
            print(f"Chosen Assignment: {' '.join(['-' if i == 0 else chr(ord('A') + i - 1) for i in chosen_assignment])}")
            print(f"Chosen Objectives: {chosen_objectives}")
            print()

            # Load the additional tasks for simulation
            with open(additional_dataset, 'r') as f:
                lines = f.readlines()
            
                additional_num_tasks = int(lines[0].strip())
                ids = np.array(list(map(int, lines[1].split())))
                rewards = np.array(list(map(int, lines[2].split())))

                conflict_sets = []
                for line in lines[3:]:
                    conflict = list(map(int, line.split()))
                    if conflict[0] == -1:
                        conflict_sets.append([])
                    else:
                        conflict_sets.append(conflict)

                additional_tasks = [{
                    "id": ids[i],
                    "reward": rewards[i],
                    "conflict_set": conflict_sets[i]
                } for i in range(additional_num_tasks)]

            solver = Solver(file_path=initial_dataset, model_path=f"{phase1_folder}/qmix_model.pth")

            # Metrics to evaluate the performance
            runtimes = []
            similarities = []

            for task in additional_tasks:
                print(f"Adding task {task['id']} with reward {task['reward']} and conflicts {task['conflict_set']}")
                
                # Solve the problem with the new task
                start_time = time.time()
                new_assignment, objective = solver.solve(current_assignment=chosen_assignment, new_task=task, ref_point=chosen_objectives, pop_size=100, n_gen=500, seed=None, verbose=False)
                end_time = time.time()

                # Calculate the runtime and similarity
                runtimes.append(end_time - start_time)
                similarities.append(1 - np.linalg.norm(objective - chosen_objectives))

                # Replace from new task's ID onwards with the new assignment
                chosen_assignment = np.append(chosen_assignment, 0)  # Append a zero to make the assignment length consistent
                chosen_assignment[task['id']:] = new_assignment
                
                # Print the results
                print(f"Result for task {task['id']}:")
                print(f"Objectives: {objective}")
                print(f"Assignments: {' '.join(['-' if i == 0 else chr(ord('A') + i - 1) for i in chosen_assignment])}")
                print()

            # Save the results
            with open(f"{results_folder}/{algo_name}.txt", 'a') as f:
                f.write(f"{np.mean(runtimes):.4f} {np.mean(similarities):.4f}\n")