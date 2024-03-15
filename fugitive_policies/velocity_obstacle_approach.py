import numpy as np
import random

class vo_local_planner:
    def __init__(self, agent_radius=1.0, obstacle_radius=1.0, time_horizon=5.0):
        self.agent_radius = agent_radius
        self.obstacle_radius = obstacle_radius
        self.time_horizon = time_horizon

    def compute_velocity_obstacle(self, agent_pos, agent_velocity, obstacle_positions, obstacle_velocities):

        def convert_angle(angle):
            # Ensure the angle is within the range [-π, π]
            angle = np.mod(angle, 2 * np.pi)
            # Convert negative angles to positive equivalent
            if angle < 0:
                angle += 2 * np.pi
            return angle

        vo = []
        for obstacle_pos, obstacle_vel in zip(obstacle_positions, obstacle_velocities):
            relative_velocity = agent_velocity - obstacle_vel
            relative_position = obstacle_pos - agent_pos
            relative_distance = np.linalg.norm(relative_position)

            # Compute the time to collision
            time_to_collision = np.dot(relative_position, relative_velocity) / np.dot(relative_velocity, relative_velocity)

            # Compute the collision distance
            collision_distance = np.linalg.norm(relative_position - time_to_collision * relative_velocity)

            # Check if a collision is possible within the time horizon
            if time_to_collision > 0 and time_to_collision < self.time_horizon and collision_distance < self.agent_radius + self.obstacle_radius:
                # Compute the velocity obstacle cone
                theta_o = np.arcsin((self.agent_radius + self.obstacle_radius) / relative_distance)
                vo_left = convert_angle(np.arctan2(relative_position[1], relative_position[0]) + theta_o)
                vo_right = convert_angle(np.arctan2(relative_position[1], relative_position[0]) - theta_o)
                vo.append((np.minimum(vo_left, vo_right), np.maximum(vo_left, vo_right)))

                if relative_distance < (self.agent_radius + self.obstacle_radius):
                    vo = []
                    break

        return vo

    def sample_safe_velocity(self, vo_cones, speed_limit):
        while True:
            # Sample a random direction
            direction = np.random.uniform(0, 2 * np.pi)
            # Sample a random speed within the limit
            speed = random.uniform(0, speed_limit)
            # Check if the norm of the sampled velocity is within the speed limit
            if speed <= speed_limit:
                # Check if the sampled velocity lies outside all velocity obstacle cones
                outside_vo = True
                for vo_cone in vo_cones:
                    vo_left, vo_right = vo_cone
                    if vo_left <= direction <= vo_right:
                        outside_vo = False
                        break
                
                if outside_vo:
                    return [np.array([speed, direction])]

if __name__ == "__main__":
    # Example usage
    agent_pos = np.array([0, 0])
    agent_velocity = np.array([1, 0])
    obstacle_positions = [np.array([3, 0])]
    obstacle_velocities = [np.array([-1, 0])]
    velocity_limits = [-2, 2]

    vo_method = vo_local_planner()
    vo_cones = vo_method.compute_velocity_obstacle(agent_velocity, obstacle_positions, obstacle_velocities)

    sampled_velocity = vo_method.sample_safe_velocity(agent_velocity, vo_cones, velocity_limits)
    print("Sampled Safe Velocity:", sampled_velocity)
