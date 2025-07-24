import gymnasium as gym
import numpy as np
import torch
from collections import deque
import cv2

class CarRacingPreprocessor:
    """Preprocesses CarRacing observations for better learning"""

    def __init__(self, img_size=(84, 84)):
        self.img_size = img_size

    def preprocess(self, obs):
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # Crop relevant area (remove score display)
        cropped = gray[0:84, 6:90]

        # Resize
        resized = cv2.resize(cropped, self.img_size)

        # Normalize
        normalized = resized.astype(np.float32) / 255.0

        return normalized


class CarRacingEnvironment:
    """Wrapper for CarRacing environment"""

    def __init__(self):
        self.env = gym.make('CarRacing-v3', render_mode=None, continuous=False)
        self.preprocessor = CarRacingPreprocessor()

        # Define discrete actions
        self.actions = [
            0,  # Do nothing
            1,  # Right
            2,  # Left
            3,  # Accelerate
            4,  # Brake
        ]

        self.frame_stack = 4
        self.frames = deque(maxlen=self.frame_stack)

    def reset(self):
        obs, _ = self.env.reset()
        processed_obs = self.preprocessor.preprocess(obs)

        # Initialize frame stack
        for _ in range(self.frame_stack):
            self.frames.append(processed_obs)

        return np.array(self.frames)

    def step(self, action_idx):
        action = self.actions[action_idx]
        obs, reward, terminated, truncated, info = self.env.step(action)

        processed_obs = self.preprocessor.preprocess(obs)
        self.frames.append(processed_obs)

        """
        # Reward shaping
        if reward < -10:  # Only penalize major off-track events
            reward = -10
        elif reward < 0:  # Normal frame penalty, keep it but maybe scale
            reward = reward * 0.5

            # Add small positive reward for staying on track
        if reward >= 0:
            reward = reward + 0.1
        """

        done = terminated or truncated
        return np.array(self.frames), reward, done, info

    def close(self):
        self.env.close()