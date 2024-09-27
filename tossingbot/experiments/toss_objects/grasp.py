import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm 
from tossingbot.utils.misc_utils import set_seed
from tossingbot.envs.pybullet.tasks import TossObjects
from tossingbot.utils.pytorch_utils import initialize_weights
from tossingbot.envs.pybullet.utils.camera_utils import plot_heightmaps
from tossingbot.agents.physics_agent import PhysicsAgent, PhysicsController
from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule

if __name__ == '__main__':
    set_seed()

    # Set device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    box_n_rows, box_n_cols = 1, 1

    r_h = 0.5
    post_grasp_h = 0.3
    n_rotations = 1
    phi_deg = 0

    total_episodes = 10_000

    # Env
    env = TossObjects(
        scene_config={
            'box_n_rows': box_n_rows,
            'box_n_cols': box_n_cols,
        },
        camera_config={'n_rotations': n_rotations})

    # Networks
    perception_module = PerceptionModule()
    grasping_module = GraspingModule()
    throwing_module = ThrowingModule()

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

    # Main loop
    obs, info = env.reset()
    for episode in tqdm(range(total_episodes), desc="Training Progress"):
        action, intermediates = agent.predict(obs, n_rotations=n_rotations, phi_deg=phi_deg)
        # plot_heightmaps(intermidiates['depth_heightmaps'])
        obs, reward, terminated, truncated, info = env.step(action=action)

        # Extract the labels from the environment's reward (grasp label and ground truth residual)
        grasp_label, gt_residual_label = reward
        
        # Initialize loss variable
        loss = 0

        # Grasping loss calculation
        y_i = torch.tensor(np.array(grasp_label)).to(device=device)
        q_i = intermediates['q_i_logits']  # logits from the prediction
        grasp_loss = grasp_criterion(q_i, y_i)  # Compute CrossEntropyLoss with logits
        loss += grasp_loss

        # Throwing loss calculation (if ground truth residual is available)
        if gt_residual_label is not None:
            delta_i_bar = torch.tensor(np.array(gt_residual_label)).to(device=device)
            delta_i = intermediates['delta_i']
            throw_loss = y_i * throw_criterion(delta_i, delta_i_bar)  # Huber loss for throwing
            loss += throw_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()