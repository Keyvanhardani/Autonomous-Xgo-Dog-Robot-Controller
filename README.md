
# Autonomous Xgo Dog Robot Controller

This repository contains the code for an autonomous robotic dog, designed to interact with its environment using object detection and distance estimation. The robot utilizes a camera to detect objects (specifically people) and calculate the distance to them, adjusting its movements accordingly.

## Features

- Object detection with a pre-trained ONNX model.
- Distance estimation using camera focal length and known object width.
- Autonomous movement control to approach or maintain a specific distance from detected people.
- Turning behavior when no objects are detected, enabling the robot to search its environment.

## Requirements

- OpenCV
- NumPy
- ONNX Runtime
- Spidev
- Xgoscreen
- Xgolib
- Pillow
- Threading
- Random

## Usage

1. Ensure the Xgo robot is connected and the camera is set up correctly.
2. Run the controller script to start the robot's autonomous behavior.
3. The robot will move towards detected people, maintaining a safe distance.
4. If no people are detected, the robot will turn to search for them.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Special thanks to the contributors of the libraries and tools used in this project.

- ## Developer

Keyvan Hardani - NOV.2023
