import os
from nsga2.nsga2 import NSGA2_solve
from qmix.qmix import QMIX
import numpy as np

if __name__ == "__main__":
    # Define the size of the dataset
    num_agents = 3
    num_tasks = 30

    # Determine the folder name based on the number of tasks
    scenario_name = "Small scale" if num_tasks < 100 else "Large scale"
    file_path = f"datasets/{scenario_name}/n{num_agents}_m{num_tasks}.txt"

    for i in range(30):
        # Define the folder to save the results
        results_folder = f"Phase1/results/{scenario_name}/n{num_agents}_m{num_tasks}/{i+1}/"
        # Create the results folder if it does not exist
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Run NSGA-II to get target assignments
        moo_res = NSGA2_solve(file_path=file_path, 
                            pop_size=80, 
                            n_gen=500, 
                            seed=None, 
                            verbose=False)

        # Print the result in table format
        print()
        print(f"No. \tSolution  \t \t \t \t \t \t \t \t[Completion Rate, Reward]")
        for i in range(len(moo_res.X)):
            print(f"{i+1}\t{' '.join(['-' if i == 0 else chr(ord('A') + i - 1) for i in moo_res.X[i]])} \t \t[{-moo_res.F[i][0]:.2f}, {-moo_res.F[i][1]:.2f}]")

        # Get the prefered assignment from the NSGA-II results
        # assignment_id = int(input(f"Enter the preferred assignment ID (1 to {len(moo_res.X)}): ")) - 1
        
        # Get the most frequent objective values
        unique_objectives, counts = np.unique(moo_res.F, axis=0, return_counts=True)
        max_count_index = np.argmax(counts)
        assignment_id = np.where((moo_res.F == unique_objectives[max_count_index]).all(axis=1))[0][0]

        target_objectives = moo_res.F[assignment_id]

        # Save the chosen assignment and its objectives into a numPy file
        np.save(f"{results_folder}/chosen_assignment.npy", moo_res.X[assignment_id])
        np.save(f"{results_folder}/chosen_objectives.npy", -target_objectives)
        
        # Extract target assignments for QMIX (all assignments with the same objectives)
        target_assignments = []
        for i in range(len(moo_res.X)):
            if np.all(moo_res.F[i] == target_objectives):
                target_assignments.append(moo_res.X[i])
        target_assignments = np.array(target_assignments)

        # Initialize and train the QMIX model
        qmix_model = QMIX(file_path=file_path,
                        target_assignments=target_assignments,
                        optimizer=None,
                        gamma=0.90,
                        verbose=True)
        
        qmix_model.train(episodes=2000)

        # Save the trained model
        qmix_model.save_model(f"{results_folder}/qmix_model.pth")