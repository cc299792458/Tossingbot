import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm 
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
    use_gui = False
    box_n_rows, box_n_cols = 1, 1

    r_h = 0.5
    post_grasp_h = 0.3
    n_rotations = 1
    phi_deg = 0

    total_episodes = 10_000

    # Env
    env = TossObjects(
        use_gui=use_gui,
        scene_config={
            'box_n_rows': box_n_rows,
            'box_n_cols': box_n_cols,
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
    for episode_num in tqdm(range(start_episode, total_episodes), desc="Training Progress"):
        action, intermediates = agent.predict([obs], n_rotations=n_rotations, phi_deg=phi_deg, episode_num=episode_num)
        next_obs, reward, terminated, truncated, info = env.step(action=action[0])

        # Store (obs, action, reward) in the replay buffer
        replay_buffer.push((obs, action, reward))

        # Update current obs
        obs = next_obs

        # If enough samples are available in the replay buffer, sample a batch
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            
            batch_loss = torch.tensor(0.0, device=device)
            
            # Process each sample in the batch
            for (obs_batch, action_batch, reward_batch) in batch:
                loss = torch.tensor(0.0, device=device)

                # Recompute intermediates using the current model
                _, intermediates = agent.predict(obs_batch, n_rotations=n_rotations, phi_deg=phi_deg, episode_num=episode_num)
                
                # Grasping loss calculation
                grasp_label, gt_residual_label = reward_batch
                y_i = torch.tensor(np.array(grasp_label), dtype=torch.long).to(device=device)
                q_i = intermediates['q_i_logits']  # logits from the stored intermediates
                grasp_loss = grasp_criterion(q_i, y_i)
                loss = loss + grasp_loss

                # Throwing loss calculation (if ground truth residual is available)
                if gt_residual_label is not None:
                    delta_i_bar = torch.tensor(np.array(gt_residual_label), dtype=torch.float).to(device=device)
                    delta_i = intermediates['delta_i']
                    throw_loss = throw_criterion(delta_i, delta_i_bar)
                    loss = loss + throw_loss

                batch_loss = batch_loss + loss

            # Backpropagation for the batch
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    # Save model after training
    save_model(agent, optimizer, episode_num, log_dir)