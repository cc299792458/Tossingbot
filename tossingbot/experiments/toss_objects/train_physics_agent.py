"""
    In this experiment, we train the PhysicsAgent to optimize the grasping of an object. 
    Since the throwing velocity is directly provided by the PhysicsController, 
    the throwing loss is not calculated and does not participate in the training process.
"""

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

def plot_success_rates(avg_grasp_success_history, avg_throw_success_history, log_dir):
    """
    Plot the grasp and throw success rates over episodes.

    Args:
        avg_grasp_success_history (list): History of average grasp success rate over time.
        avg_throw_success_history (list): History of average throw success rate over time.
        log_dir: Directory to save the plot
    """
    # Episodes range
    episodes = range(len(avg_grasp_success_history))

    # Plotting
    plt.figure(figsize=(10, 5))

    plt.plot(episodes, avg_grasp_success_history, label='Average Grasp Success Rate', color='blue')
    plt.plot(episodes, avg_throw_success_history, label='Average Throw Success Rate', color='green')

    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title('Grasp and Throw Success Rates Over Time')
    plt.legend(loc='best')
    plt.grid(True)

    plot_path = os.path.join(log_dir, 'residual_physics_agent_training_curve.png')
    plt.savefig(plot_path)

    plt.show()

if __name__ == '__main__':
    set_seed()

    # Log directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs") 
    
    # Set device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If load model or not
    load_model_ = False

    # Parameters
    use_gui = False
    box_length = 0.15
    box_n_rows, box_n_cols = 3, 3
    use_heuristic = False

    n_rotations = 1
    phi_deg = 45

    total_episodes = 2_000

    # Lists to track the cumulative success rates over all episodes
    avg_grasp_success_history = []
    avg_throw_success_history = []

    # Create deque to track the recent history of grasp and throw successes (for short-term stats)
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
        task_config={
            'use_heuristic': use_heuristic,
        },
        objects_config={"object_types": ['ball', 'cube', 'rod']},
        camera_config={'n_rotations': n_rotations},
    )

    # Networks
    perception_module = PerceptionModule()
    grasping_module = GraspingModule()
    throwing_module = ThrowingModule()

    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=10000)
    batch_size = 32  # Set batch size

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
        epsilons=[0.5, 0.1],
        total_episodes=total_episodes,
    )

    # Optimizer
    optimizer = optim.Adam(agent.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=2e-5)

    # Loss functions
    grasp_criterion = nn.CrossEntropyLoss()

    # Optionally load the model
    if not load_model_:
        start_episode = 0
    else:
        start_episode = load_model(agent, optimizer, log_dir, 'phyiscs_agent')

    # Main loop
    obs, info = env.reset()
    progress_bar = tqdm(range(start_episode, total_episodes), desc="Training Progress")

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

        # Record grasp success and throw success for this episode (for both recent and full history)
        grasp_success_history.append(info['grasp_success'])

        if info['grasp_success']:
            throw_success_history.append(info['throw_success'])

        # Calculate short-term average success rates based on the recent history
        avg_grasp_success = np.mean(grasp_success_history)
        avg_throw_success = np.mean(throw_success_history) if throw_success_history else 0.0

        # Append the cumulative success rates to the history for plotting
        avg_grasp_success_history.append(avg_grasp_success)
        avg_throw_success_history.append(avg_throw_success)

        # Update the tqdm description to display the short-term success rates
        progress_bar.set_postfix({
            "Grasp Success": f"{avg_grasp_success:.3f}",
            "Throw Success": f"{avg_throw_success:.3f}"
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
            
            q_i, _ = agent.extract_logits_for_loss(
                q_g=intermediates['q_g'], 
                q_t=intermediates['q_t'], 
                grasp_pixel_indices=np.array(grasp_pixel_indices_batch)
            )

            # Grasping loss calculation
            y_i = torch.tensor(np.array(grasp_label_batch), dtype=torch.long).to(device=device)
            grasp_loss = grasp_criterion(q_i, y_i)
            loss = loss + grasp_loss

            # Backpropagation for the batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Plot cumulative success rates after training
    plot_success_rates(avg_grasp_success_history, avg_throw_success_history, log_dir)

    # Save model after training
    save_model(agent, optimizer, episode_num, log_dir, model_name='physics_agent')