# Tossingbot
This repository is used to **replicate** the TossingBot(https://tossingbot.cs.princeton.edu/) implementation.

## Simulation

Simulation frequency: 240Hz

## Robot

Control frequency: 60Hz

Robot type: Franka Panda

### Motion Primitives

#### Grasping Primitive

#### Throwing Primitive

## Task

### Workspace

Compared to paper's 0.9 × 0.7m workspace, I employ a relatively smaller workspace, which is 0.4 * 0.3m.

### Bins (target boxes)

### Objects

Currently, **ball, cube, rod, and hammer** are supported. [Refer here](tossingbot/envs/pybullet/utils/objects_utils.py) for more details.

![Objects](https://github.com/cc299792458/Tossingbot/blob/main/images/objects.png)

## Experiments
