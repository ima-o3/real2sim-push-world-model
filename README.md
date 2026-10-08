# Real-to-Sim Push World Model

Learning tabletop pushing dynamics from real-world videos and transferring actions to a simulated Franka Panda robot.

## Project Pipeline

Real video ? OpenCV tracking ? Action extraction ? MuJoCo Panda ? Object response

## Technology

- Python
- OpenCV
- MuJoCo
- Franka Panda
- Jacobian-based inverse kinematics

## Current Progress

- [x] MuJoCo Panda simulation
- [x] Physical object pushing
- [x] Three real-world pilot videos recorded
- [x] P01 video trajectory extraction
- [x] P01 joint-space replay
- [x] P01 Cartesian trajectory replay
- [ ] Validate stick-like end effector
- [ ] Replay P02 and P03
- [ ] Train and evaluate world model
- [ ] Final comparison and demonstration

## Experimental Results

P01 Cartesian replay:
- Real pusher net displacement: 133.7 mm
- Actual simulated end-effector path: 154.3 mm
- Cartesian tracking RMSE: 9.14 mm
- Simulated object displacement: 45.2 mm
- Real object displacement: 145.3 mm

The remaining discrepancy is under investigation.

## Reproducibility

Setup instructions, required dependencies and experiment commands will be finalised before submission.
