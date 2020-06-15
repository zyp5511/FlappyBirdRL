# FlappyBirdRL
DQN applied to Flappy Bird with pygame and PyTorch.

* `FlappyBird_original/train.py` runs the baseline
* `FlappyBird_original/double_train.py` runs the Double DQN version
* `FlappyBird_original/double_train_pr.py` runs the Double DQN with Prioritized Replay
* `FlappyBird_original/inference.py` runs inference of a trained model
* `FlappyBird_reward_shaping/*train*.py` runs the aforementioned algorithm with the modified reward that quantifies the closeness of the bird to the target pipe gap when it crashes a pipe.

# Credit:
1. https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch
