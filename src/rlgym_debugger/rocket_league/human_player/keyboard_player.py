import keyboard
import mouse
import numpy as np

from rlgym_debugger.api.human_player.player import DeviceInterface

from rlgym.rocket_league.common_values import (
    THROTTLE,
    PITCH,
    STEER,
    YAW,
    ROLL,
    BOOST,
    JUMP,
    HANDBRAKE,
)

BINDS = {
    "THROTTLE_UP": "w",
    "THROTTLE_DOWN": "s",
    "YAW_RIGHT": "d",
    "YAW_LEFT": "a",
    "BOOST": "right",
    "JUMP": "left",
    "PITCH_UP": "s",
    "PITCH_DOWN": "w",
    "ROLL": "left shift",
    "POWERSLIDE": "left shift",
}


class KeyboardInterface(DeviceInterface[np.ndarray]):
    def get_output(self) -> np.ndarray:
        _final_actions = np.zeros((1, 8))

        _throttle = 0
        if keyboard.is_pressed(BINDS["THROTTLE_UP"]):
            _throttle += 1
        if keyboard.is_pressed(BINDS["THROTTLE_DOWN"]):
            _throttle -= 1

        _pitch = 0
        if keyboard.is_pressed(BINDS["PITCH_UP"]):
            _pitch += 1
        if keyboard.is_pressed(BINDS["PITCH_DOWN"]):
            _pitch -= 1

        _yaw = 0
        if keyboard.is_pressed(BINDS["YAW_RIGHT"]):
            _yaw += 1
        if keyboard.is_pressed(BINDS["YAW_LEFT"]):
            _yaw -= 1

        _jump = mouse.is_pressed(BINDS["JUMP"])
        _boost = mouse.is_pressed(BINDS["BOOST"])
        _roll, _powerslide = (
            keyboard.is_pressed(BINDS["ROLL"]),
            keyboard.is_pressed(BINDS["POWERSLIDE"]),
        )

        _final_actions[:, THROTTLE] = _throttle
        _final_actions[:, STEER] = _yaw
        _final_actions[:, PITCH] = _pitch
        _final_actions[:, YAW] = _yaw * (not _roll)
        _final_actions[:, JUMP] = _jump
        _final_actions[:, BOOST] = _boost
        _final_actions[:, ROLL] = _yaw * _roll
        _final_actions[:, HANDBRAKE] = _powerslide

        return _final_actions
