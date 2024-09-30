import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from collections import deque
from tossingbot.utils.misc_utils import set_seed
from tossingbot.envs.pybullet.tasks import TossObjects
from tossingbot.envs.pybullet.utils.camera_utils import plot_heightmaps
from tossingbot.agents.physics_agent import PhysicsAgent, PhysicsController
from tossingbot.utils.pytorch_utils import initialize_weights, load_model, save_model
from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule, ReplayBuffer

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    set_seed()

    # Log directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs") 
    
    # Set device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    use_gui = True
    box_n_rows, box_n_cols = 1, 1
    n_object = 1

    r_h = 0.5
    post_grasp_h = 0.3
    n_rotations = 1
    phi_deg = 0

    total_episodes = 10_000

    # Create deque to track the history of grasp and throw successes
    grasp_success_history = deque(maxlen=1000)
    throw_success_history = deque(maxlen=1000)

    # Env
    env = TossObjects(
        use_gui=use_gui,
        scene_config={
            'box_n_rows': box_n_rows,
            'box_n_cols': box_n_cols,
        },
        objects_config={
            'n_object': n_object,
        },
        camera_config={'n_rotations': n_rotations}
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
    physics_controller = PhysicsController(r_h=r_h)
    agent = PhysicsAgent(
        device=device, 
        perception_module=perception_module, 
        grasping_module=grasping_module, 
        throwing_module=throwing_module,
        physics_controller=physics_controller,
        post_grasp_h=post_grasp_h,
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
    for episode_num in progress_bar:
        action, intermediates = agent.predict([obs], n_rotations=n_rotations, phi_deg=phi_deg, episode_num=episode_num)
        next_obs, reward, terminated, truncated, info = env.step(action=action[0])

        # Store (obs, action, reward) in the replay buffer
        replay_buffer.push((obs, action[0], reward))

        # Update current obs
        obs = next_obs

        # Record grasp success and throw success for this episode
        grasp_success_history.append(info['grasp_success'])
        throw_success_history.append(info['throw_success'])

        # Compute average success rates over the last 1000 episodes (or fewer if not enough episodes)
        avg_grasp_success = np.mean(grasp_success_history)
        avg_throw_success = np.mean(throw_success_history)

        # Update the tqdm description to display the success rates
        progress_bar.set_postfix({
            "Grasp Success Rate": f"{avg_grasp_success:.3f}",
            "Throw Success Rate": f"{avg_throw_success:.3f}"
        })

        # If enough samples are available in the replay buffer, sample a batch
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            
            batch_loss = torch.tensor(0.0, device=device)
            
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

            # Throwing loss calculation (if ground truth residual is available)
            for (b, gt_residual_label) in enumerate(gt_residual_label_batch):
                if gt_residual_label is not None:
                    delta_i_bar = torch.tensor(np.array(gt_residual_label), dtype=torch.float).to(device=device)
                    throw_loss = throw_criterion(delta_i[b], delta_i_bar)
                    loss = loss + throw_loss

            batch_loss = batch_loss + loss

            # Backpropagation for the batch
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    # Save model after training
    save_model(agent, optimizer, episode_num, log_dir)
