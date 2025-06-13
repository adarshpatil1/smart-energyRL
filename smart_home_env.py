import numpy as np

class SmartHomeEnv:
    def __init__(self):
        self.action_space = type('', (), {})()  # Create an empty object
        self.action_space.n = 4  # Or whatever number of actions you have
        self.hour = 0
        self.temp_outside = 25
        self.devices = {
            "heater": False,
            "light": False,
            "fan": False
        }
        self.state = self._get_state()

    def _get_state(self):
        return np.array([
            self.hour,
            int(self.devices["heater"]),
            int(self.devices["light"]),
            int(self.devices["fan"]),
            self.temp_outside
        ])

    def step(self, action):
        if action == 1:
            self.devices["heater"] = not self.devices["heater"]
        elif action == 2:
            self.devices["light"] = not self.devices["light"]
        elif action == 3:
            self.devices["fan"] = not self.devices["fan"]

        self.hour = (self.hour + 1) % 24
        self.temp_outside += np.random.normal(0, 0.5)

        reward = self._get_reward()
        self.state = self._get_state()
        done = self.hour == 23

        return self.state, reward, done, {}

    def reset(self):
        self.__init__()
        return self.state

    def _get_reward(self):
        energy_cost = 0
        if self.devices["heater"]: energy_cost += 3
        if self.devices["light"]: energy_cost += 2
        if self.devices["fan"]: energy_cost += 1

        comfort_penalty = 0
        if self.devices["heater"] and self.temp_outside > 28:
            comfort_penalty += 2
        if not self.devices["light"] and self.hour >= 19:
            comfort_penalty += 1

        return - (energy_cost + comfort_penalty)
