from .environment import CustomEnv
from .models import AgentQNetwork, MixingNetwork
import numpy as np
import torch
import torch.optim as optim
from queue import Queue
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 30})

class QMIX:
    def __init__(self, file_path, target_assignments=None, optimizer=None, gamma=0.90, memory_length=20, verbose=False):
        self.env = CustomEnv(file_path=file_path, target_assignments=target_assignments)
        self.memory_length = memory_length  # Length of the memory for each agent 
        self.agent_qs = [AgentQNetwork(2 * self.memory_length + 1) for _ in range(self.env.num_agents)] # List of Q-networks for each agent
        self.mixing_net = MixingNetwork(self.env.num_agents, self.memory_length + 1)  # Mixing network to combine Q-values
        params = list(self.mixing_net.parameters()) + [p for a in self.agent_qs for p in a.parameters()] # Params including all agent Q-networks and the mixing network
        self.optimizer = optimizer if optimizer is not None else optim.Adam(params, lr=1e-3) # Optimizer for the networks
        self.gamma = gamma  # Discount factor for future rewards
        self.verbose = verbose  # Verbose output for training progress

    # Training so the QMIX can generate the target assignments (offline phase)
    def train(self, episodes):
        for ep in range(episodes):
            # Initialize state and agent memories
            state = self.env.reset()
            done = False
            task_index = 0
            agent_memory = [Queue() for _ in range(self.env.num_agents)]
            for q in agent_memory:
                for _ in range(self.memory_length):
                    q.put(0)

            total_reward = 0
            while not done:
                # Get the local observation for each agent
                reward, conflict = self.env.get_task_info(task_index)
                conflict = conflict[max(0, task_index - self.memory_length):task_index]
                conflict = np.pad(conflict, (self.memory_length - len(conflict), 0), mode='constant', constant_values=0)

                obs = [
                    torch.tensor(list(agent_memory[i].queue) + list(conflict) + [reward], dtype=torch.float32)
                    for i in range(self.env.num_agents)
                ]

                # Forward pass through the agent Q-networks and the mixing network
                q_vals = torch.stack([self.agent_qs[i](obs[i]) for i in range(self.env.num_agents)])
                global_state = state[max(0, task_index - self.memory_length):task_index]
                global_state = np.pad(global_state, (self.memory_length - len(global_state), 0), mode='constant', constant_values=0)
                q_tot = self.mixing_net(q_vals, torch.tensor(list(global_state) + [reward], dtype=torch.float32))
                
                # Choose an agent based on the softmax probabilities of the Q-values
                chosen_agent = np.random.choice(self.env.num_agents, p=q_tot.detach().numpy().flatten())

                next_state, rewards, done = self.env.step(task_index, chosen_agent)
                total_reward += np.sum(rewards)

                # Update the agent memories, chosen_agent is 1-indexed (0 is no agent), so we adjust it to 0-indexed
                for i in range(self.env.num_agents):
                    agent_memory[i].get()
                    agent_memory[i].put(1 if i == next_state[task_index] - 1 else 0)

                # If the task assignment process is not done, calculate the next Q-values
                if not done:
                    with torch.no_grad():
                        next_reward, next_conflict = self.env.get_task_info(task_index + 1)
                        next_conflict = next_conflict[max(0, task_index + 1 - self.memory_length):task_index + 1]
                        next_conflict = np.pad(next_conflict, (self.memory_length - len(next_conflict), 0), mode='constant', constant_values=0)

                        next_obs = [
                            torch.tensor(list(agent_memory[i].queue) + list(next_conflict) + [next_reward], dtype=torch.float32)
                            for i in range(self.env.num_agents)
                        ]

                        next_q_vals = torch.stack([self.agent_qs[i](next_obs[i]) for i in range(self.env.num_agents)])
                        next_global_state = next_state[max(0, task_index + 1 - self.memory_length):task_index + 1]
                        next_global_state = np.pad(next_global_state, (self.memory_length - len(next_global_state), 0), mode='constant', constant_values=0)
                        next_q_tot = self.mixing_net(next_q_vals, torch.tensor(list(next_global_state) + [reward], dtype=torch.float32))
                else:
                    next_q_tot = torch.ones_like(q_tot)

                # Calculate the target Q-values
                y = torch.tensor(rewards, dtype=torch.float32) + self.gamma * torch.max(next_q_tot) * (rewards > 0)
                loss = (q_tot - y).pow(2).mean()

                # Update the networks
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # State transition
                state = next_state
                task_index += 1

            if self.verbose and (ep + 1) % 100 == 0:
                print(f"Episode {ep+1}:\t Reward = {total_reward}")

    # Rescheduling the tasks from the new task onward based on the trained QMIX model (online phase)
    def plan(self, current_assignment, new_task, **kwargs):
        # Update the environment with the new task
        self.env.update(new_task)
        
        # Set the current state based on the current assignment
        self.env.reset()
        self.env.state[:new_task['id']] = current_assignment[:new_task['id']]  # Keep the current assignment for completed tasks

        # Initialize agent memories
        agent_memory = [Queue() for _ in range(self.env.num_agents)]
        for q in agent_memory:
            for _ in range(self.memory_length):
                q.put(0)
        for agent_id in current_assignment[:new_task['id']]:
            for i in range(self.env.num_agents):
                agent_memory[i].get()
                agent_memory[i].put(1 if i == agent_id - 1 else 0) # Agent IDs are 1-indexed, so we adjust it to 0-indexed

        # Initialize the task index
        task_index = new_task['id']
        done = False
        state = self.env.state.copy()

        while not done:
            # Get the local observation for each agent
            reward, conflict = self.env.get_task_info(task_index)
            conflict = conflict[max(0, task_index - self.memory_length):task_index]
            conflict = np.pad(conflict, (self.memory_length - len(conflict), 0), mode='constant', constant_values=0)

            obs = [
                torch.tensor(list(agent_memory[i].queue) + list(conflict) + [reward], dtype=torch.float32)
                for i in range(self.env.num_agents)
            ]

            # Forward pass through the agent Q-networks and the mixing network
            q_vals = torch.stack([self.agent_qs[i](obs[i]) for i in range(self.env.num_agents)])
            global_state = state[max(0, task_index - self.memory_length):task_index]
            global_state = np.pad(global_state, (self.memory_length - len(global_state), 0), mode='constant', constant_values=0)
            q_tot = self.mixing_net(q_vals, torch.tensor(list(global_state) + [reward], dtype=torch.float32))

            # Choose an agent based on the Q-values
            decisions = np.argsort(q_tot.detach().numpy().flatten())[::-1]  # Sort Q-values in descending order
            for chosen_agent in decisions:
                next_state, _, done = self.env.step(task_index, chosen_agent)
                if self.env.check_valid_state():
                    break

            # Update agent memories
            for i in range(self.env.num_agents):
                agent_memory[i].get()
                agent_memory[i].put(1 if i == chosen_agent - 1 else 0)

            state = next_state
            task_index += 1

        return state[new_task['id']:], self.env.evaluate()  # Return the assignment for the new task and the evaluation of the assignment

    def save_model(self, file_path):
        torch.save({
            'agent_qs': [agent.state_dict() for agent in self.agent_qs],
            'mixing_net': self.mixing_net.state_dict(),
        }, file_path)

    def load_model(self, file_path):
        checkpoint = torch.load(file_path)
        for i, agent in enumerate(self.agent_qs):
            agent.load_state_dict(checkpoint['agent_qs'][i])
        self.mixing_net.load_state_dict(checkpoint['mixing_net'])
        if self.verbose:
            print(f"Model loaded from {file_path}")