import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import deque
from tossingbot.utils.misc_utils import set_seed
from tossingbot.envs.pybullet.tasks import TossObjects
from tossingbot.agents.physics_agent import PhysicsAgent, PhysicsController
from tossingbot.utils.pytorch_utils import initialize_weights, load_model, save_model
from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule, ReplayBuffer

torch.autograd.set_detect_anomaly(True)

import matplotlib.pyplot as plt
import numpy as np

def plot_success_rates(grasp_success_history, throw_success_history, total_episodes):
    """
    Plot the grasp success and throw success rates over episodes.

    Args:
        grasp_success_history (deque or list): History of grasp success (1 for success, 0 for failure).
        throw_success_history (deque or list): History of throw success (1 for success, 0 for failure).
        total_episodes (int): Total number of episodes.
    """
    # Convert deque to list for slicing
    grasp_success_history = list(grasp_success_history)
    throw_success_history = list(throw_success_history)

    # Episodes range
    episodes = range(len(grasp_success_history))

    # Moving average over the last 10 episodes for grasp and throw success
    avg_grasp_success = [np.mean(grasp_success_history[max(0, i-10):i+1]) for i in episodes]
    avg_throw_success = [np.mean(throw_success_history[max(0, i-10):i+1]) for i in episodes]

    # Plotting
    plt.figure(figsize=(10, 5))

    plt.plot(episodes, avg_grasp_success, label='Grasp Success Rate (Last 10)', color='blue')
    plt.plot(episodes, avg_throw_success, label='Throw Success Rate (Last 10)', color='green')

    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title('Grasp and Throw Success Rates Over Episodes')
    plt.legend(loc='best')

    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    set_seed()

    # Log directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs") 
    
    # Set device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    use_gui = False
    box_length = 0.15
    box_n_rows, box_n_cols = 3, 3

    n_rotations = 1
    phi_deg = 45

    total_episodes = 100

    # Create deque to track the history of grasp and throw successes
    grasp_success_history = deque(maxlen=10)
    throw_success_history = deque(maxlen=10)

    # Env
    env = TossObjects(
        use_gui=use_gui,
        scene_config={
            'box_n_rows': box_n_rows,
            'box_n_cols': box_n_cols,
            'box_length': box_length,
        },
        camera_config={'n_rotations': n_rotations}
    )

    # Networks
    perception_module = PerceptionModule()
    grasping_module = GraspingModule()
    throwing_module = ThrowingModule()

    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=10000)
    batch_size = 1  # Set batch size

    # Initialize weights with Xavier
    initialize_weights(perception_module)
    initialize_weights(grasping_module)
    initialize_weights(throwing_module)

    # Agent
    physics_controller = PhysicsController()
    agent = PhysicsAgent(
        device=device, 
        perception_module=perception_module, 
        grasping_module=grasping_module, 
        throwing_module=throwing_module,
        physics_controller=physics_controller,
        epsilons=[0.0, 0.0] # disable epsilon-greedy
    )

    # Optimizer
    optimizer = optim.Adam(agent.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=2e-5)

    # Loss functions
    grasp_criterion = nn.CrossEntropyLoss()
    throw_criterion = nn.SmoothL1Loss()  # Huber loss

    # Optionally load the model
    start_episode = load_model(agent, optimizer, log_dir)

    # Main loop
    obs, info = env.reset()
    progress_bar = tqdm(range(start_episode, total_episodes), desc="Training Progress")
    avg_grasp_success = 0.0
    avg_throw_success = 0.0
    for episode_num in progress_bar:
        action, intermediates = agent.predict(
            [obs], 
            n_rotations=n_rotations, 
            phi_deg=phi_deg, 
            episode_num=episode_num, 
            use_heuristic=info['next_grasp_with_heuristic']
        )
        next_obs, reward, terminated, truncated, info = env.step(action=action[0])

        # Store (obs, action, reward) in the replay buffer
        replay_buffer.push((obs, action[0], reward))

        # Update current obs
        obs = next_obs

        # Record grasp success and throw success for this episode
        grasp_success_history.append(info['grasp_success'])
        if info['grasp_success']:
            throw_success_history.append(info['throw_success'])

        # Compute average success rates over the last 10 episodes (or fewer if not enough episodes)
        avg_grasp_success = np.mean(grasp_success_history)
        if info['grasp_success']:
            avg_throw_success = np.mean(throw_success_history)

        # Update the tqdm description to display the success rates
        progress_bar.set_postfix({
            "Grasp Success Rate": f"{avg_grasp_success:.3f}",
            "Throw Success Rate": f"{avg_throw_success:.3f}"
        })

        # If enough samples are available in the replay buffer, sample a batch
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            
            # Process each sample in the batch
            obs_batch, action_batch, reward_batch = replay_buffer.unpack_batch(batch)
            grasp_label_batch, gt_residual_label_batch = replay_buffer.unpack_batch(reward_batch)
            grasp_pixel_indices_batch, _, _, _ = replay_buffer.unpack_batch(action_batch)
            
            loss = torch.tensor(0.0, device=device)

            # Recompute intermediates using the current model
            _, intermediates = agent.predict(obs_batch, n_rotations=n_rotations, phi_deg=phi_deg, episode_num=episode_num)
            
            q_i, delta_i = agent.extract_logits_for_loss(
                q_g=intermediates['q_g'], 
                q_t=intermediates['q_t'], 
                grasp_pixel_indices=np.array(grasp_pixel_indices_batch)
            )

            # Grasping loss calculation
            y_i = torch.tensor(np.array(grasp_label_batch), dtype=torch.long).to(device=device)
            grasp_loss = grasp_criterion(q_i, y_i)
            loss = loss + grasp_loss

            # # Throwing loss calculation (if ground truth residual is available)
            # for (b, gt_residual_label) in enumerate(gt_residual_label_batch):
            #     if gt_residual_label is not None:
            #         delta_i_bar = torch.tensor(np.array(gt_residual_label), dtype=torch.float).to(device=device)
            #         throw_loss = throw_criterion(delta_i[b], delta_i_bar)
            #         loss = loss + throw_loss

            # Backpropagation for the batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # # Save model after training
    # save_model(agent, optimizer, episode_num, log_dir)

    # Plot success rates after training
    plot_success_rates(grasp_success_history, throw_success_history, total_episodes)
