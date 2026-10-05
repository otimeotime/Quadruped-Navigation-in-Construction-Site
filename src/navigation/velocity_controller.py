import math
import numpy as np


# Class: Velocity command that drives the robot toward a target point
# A heading controller: the yaw rate turns the robot to face the target, and the forward
# speed fades linearly with the heading error, down to turning in place when the target
# lies turn_in_place_angle or more off the nose. When the target is the final goal, the
# speed also ramps down with the distance so the robot stops on it instead of overshooting.
# The output is a body-frame command (lin_vel_x, lin_vel_y, ang_vel) for the walking policy;
# lin_vel_y stays 0 since walking forward is what the policy tracks best.
class VelocityController:
    # Constructor
    # max_lin_vel_x: Top forward speed in m/s (the policy is trained up to 3 m/s; obstacles call for less)
    # max_ang_vel: Top yaw rate in rad/s (the policy is trained up to 1 rad/s)
    # heading_gain: Yaw rate per radian of heading error, in 1/s
    # stopping_gain: Forward speed per meter of distance left to the final goal, in 1/s
    # turn_in_place_angle: Heading error beyond which the robot stops walking and only turns, in degrees
    def __init__(self, max_lin_vel_x=1.0, max_ang_vel=1.0, heading_gain=2.0, stopping_gain=1.5,
                 turn_in_place_angle=30.0):
        self.max_lin_vel_x = max_lin_vel_x
        self.max_ang_vel = max_ang_vel
        self.heading_gain = heading_gain
        self.stopping_gain = stopping_gain
        self.turn_in_place_angle = math.radians(turn_in_place_angle)

    # Command (lin_vel_x, lin_vel_y, ang_vel) from the robot's position (x, y) and yaw (radians)
    # toward target (x, y); is_goal ramps the speed down for a stop on the target.
    def compute(self, position, yaw, target, is_goal=False):
        dx, dy = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
        # Target in the body frame
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        left = -math.sin(yaw) * dx + math.cos(yaw) * dy
        heading_error = math.atan2(left, forward)

        ang_vel = float(np.clip(self.heading_gain * heading_error, -self.max_ang_vel, self.max_ang_vel))
        lin_vel_x = self.max_lin_vel_x * max(0.0, 1.0 - abs(heading_error) / self.turn_in_place_angle)
        if is_goal:
            lin_vel_x = min(lin_vel_x, self.stopping_gain * math.hypot(dx, dy))
        return np.array([lin_vel_x, 0.0, ang_vel])
