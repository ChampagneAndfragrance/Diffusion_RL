"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

show_animation = True

class dwa_local_planner(object):
    def __init__(self, max_speed=15.0, min_speed=0, max_yaw_rate=math.pi, max_accel=3.0, max_delta_yaw_rate=np.inf, 
                    v_resolution=0.1, yaw_rate_resolution=0.15*math.pi, dt=1, predict_time=8.0, to_goal_cost_gain=0.3, 
                        speed_cost_gain=0.1, globalPath_cost_gain=0.01, obstacle_cost_gain=0.000000001, robot_stuck_flag_cons=0.001) -> None:
        # robot parameter
        self.max_speed = max_speed  # [m/s]
        self.min_speed = min_speed  # [m/s]
        self.max_yaw_rate = max_yaw_rate  # [rad/s]
        self.max_accel = max_accel  # [m/ss]
        self.max_delta_yaw_rate = max_delta_yaw_rate  # [rad/ss]
        self.v_resolution = v_resolution  # [m/s]
        self.yaw_rate_resolution = yaw_rate_resolution  # [rad/s]
        self.dt = dt  # [s] Time tick for motion prediction
        self.predict_time = predict_time  # [s]
        self.to_goal_cost_gain = to_goal_cost_gain
        self.speed_cost_gain = speed_cost_gain
        self.globalPath_cost_gain = globalPath_cost_gain
        self.obstacle_cost_gain = obstacle_cost_gain
        self.robot_stuck_flag_cons = robot_stuck_flag_cons  # constant to prevent robot stucked
        self.ob_cam = [[1600, 1800, 150]]
        self.ob_hc_sp = [[], []]
        self.ob_id = [-1] # -1 means the mountain
        self.global_path = None

    def reset(self):
        self.ob_cam = [[1600, 1800, 150]]
        self.ob_hc_sp = [[], []]
        self.ob_id = [-1] # -1 means the mountain
        self.global_path = []

    def add_to_obstacle(self, obstacles, obstacle_id_seen, blue_paths):
        # INFO: Process the dynamic agents every local plan starts or detection
        for dyna_ob_id in range(2):
            self.ob_hc_sp[dyna_ob_id] = blue_paths[dyna_ob_id]
        for obstacle, obstacle_id in zip(obstacles, obstacle_id_seen):
            # if obstacle_id < 2:
            #     self.ob_hc_sp[obstacle_id] = blue_paths[obstacle_id]
            # INFO: Process static cams and mountain
            if obstacle_id not in self.ob_id and not obstacle_id < 2:
                self.ob_cam.append(obstacle)
                self.ob_id.append(obstacle_id)
            else:
                pass
        return 

    def update_global_path(self, global_path):
        self.global_path = global_path
        pass
        return

    def dwa_control(self, x, goal, global_path, heli_path, sp_path):
        """
        Dynamic Window Approach control
        """
        dw = self.calc_dynamic_window(x)

        u, trajectory = self.calc_control_and_trajectory(x, dw, goal, global_path)

        self.plot_best_traj(trajectory, heli_path, sp_path)
        
        return u, trajectory

    def plot_best_traj(self, traj, heli_path, sp_path):
        plt.plot(self.global_path[:, 0], self.global_path[:, 1], "-r", linewidth=2, ms=5)
        plt.plot(heli_path[:, 0], heli_path[:, 1], "-bo", linewidth=2, ms=5)
        plt.plot(sp_path[:, 0], sp_path[:, 1], "-co", linewidth=2, ms=5)
        plt.plot(traj[:, 0], traj[:, 1], "-go", linewidth=2, ms=5)
        plt.show()

    def calc_dynamic_window(self, x):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [self.min_speed, self.max_speed,
            -self.max_yaw_rate, self.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [x[3] - self.max_accel * self.dt,
            x[3] + self.max_accel * self.dt,
            x[4] - self.max_delta_yaw_rate * self.dt,
            x[4] + self.max_delta_yaw_rate * self.dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def calc_control_and_trajectory(self, x, dw, goal, global_path):
        """
        calculation final input with dynamic window
        """

        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])
        static_obstacles_xy = self.ob_hc_sp[0] + self.ob_hc_sp[1] + self.ob_cam 
        static_obstacles_xy = [ele for ele in static_obstacles_xy if ele != []]
        dynamic_obstacles_xy = self.ob_hc_sp
        dynamic_obstacles_xy = [ele for ele in dynamic_obstacles_xy if ele != []]

        # evaluate all trajectory with sampled input in dynamic window
        best_traj_cost = [0, 0, 0, 0, 0]
        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for y in np.arange(dw[2], dw[3], self.yaw_rate_resolution):

                trajectory = self.predict_trajectory(x_init, v, y)
                # calc cost
                to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal) * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.speed_cost_gain * (self.max_speed - np.abs(trajectory[-1, 3]))
                static_ob_cost = 0 if len(static_obstacles_xy)==0 else self.obstacle_cost_gain * self.calc_static_obstacle_cost(trajectory, np.array(static_obstacles_xy))
                dynamic_ob_cost = 0 if len(dynamic_obstacles_xy)==0 else self.obstacle_cost_gain * self.calc_dynamic_obstacle_cost(trajectory, np.array(dynamic_obstacles_xy))
                to_global_path_cost = self.globalPath_cost_gain * self.calc_globalPath_cost(trajectory, global_path)

                # INFO: cost hard control
                dynamic_ob_cost = 0

                final_cost = to_goal_cost + speed_cost + static_ob_cost + dynamic_ob_cost + to_global_path_cost

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    best_traj_cost = [to_goal_cost, speed_cost, static_ob_cost, dynamic_ob_cost, to_global_path_cost]
                    if abs(best_u[0]) < self.robot_stuck_flag_cons \
                            and abs(x[3]) < self.robot_stuck_flag_cons:
                        # to ensure the robot do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -self.max_delta_yaw_rate
        print("The final to_goal_cost, speed_cost, static_ob_cost, dynamic_ob_cost, to_global_path_cost are: \n ", best_traj_cost[0] , "\n", best_traj_cost[1] , 
                                                                                        "\n", best_traj_cost[2], "\n", best_traj_cost[3], "\n", best_traj_cost[4] , "\n")
        return best_u, best_trajectory

    def predict_trajectory(self, x_init, v, y):
        """
        predict trajectory with an input
        """

        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time < self.predict_time:
            x = motion(x, [v, y], self.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.dt

        return trajectory

    def calc_static_obstacle_cost(self, trajectory, ob):
        """
        calc obstacle cost inf: collision
        """
        ox = ob[:, 0]
        oy = ob[:, 1]
        orange = np.expand_dims(ob[:, 2], axis=1)
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)
        # INFO: The prisoer should not collide with known camera and mountain, set cost to inf if collison happens
        # inf for mountain only
        # if np.array(r[0,:] <= orange[0,:]).any():
        #     return float("Inf")
        # inf for both
        # INFO: Set the cost to inf if the agent goes into the obstacle range, but NOT if it is in the obstacle range at the first step
        if (r[:,0] <= orange[0]).any():
            pass
        elif np.array(r <= orange).any():
            return float("Inf")
        min_r = np.min(r)
        return 1.0 / min_r  # OK

    def calc_dynamic_obstacle_cost(self, trajectory, ob):
        """
        calc obstacle cost inf: collision
        """
        dyna_ob_locs = ob[..., :2]
        dyna_ob_range = ob[..., -1:]
        traj_locs = trajectory[:, :2]
        dxy = traj_locs - dyna_ob_locs
        r = np.linalg.norm(dxy, axis=-1, keepdims=True)
        if (r[:,0] <= dyna_ob_range[:,0]).any():
            pass
        elif np.array(r <= dyna_ob_range).any():
            return float("Inf")
        min_r = np.min(r)
        return 1.0 / min_r  # OK

    def calc_globalPath_cost(self, trajectory, global_path):
        traj_end_xy = trajectory[-1][:2]
        min_dist_from_traj_end_to_globalPath = np.min(np.linalg.norm(global_path - traj_end_xy, axis=-1))
        return min_dist_from_traj_end_to_globalPath

    def calc_to_goal_cost(self, trajectory, goal):
        """
            calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 360.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 2.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.globalPath_cost_gain = 0.15
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = np.array([[-1, -1],
                            [0, 2],
                            [4.0, 2.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [5.0, 9.0],
                            [8.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 10.0],
                            [9.0, 11.0],
                            [12.0, 13.0],
                            [12.0, 12.0],
                            [15.0, 15.0],
                            [13.0, 13.0]
                            ])

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Config()


def motion(x, u, dt):
    """
    motion model
    """

    # INFO: For rigid body
    # x[2] += u[1] * dt
    # x[0] += u[0] * math.cos(x[2]) * dt
    # x[1] += u[0] * math.sin(x[2]) * dt
    # INFO: For particle
    x[0] += u[0] * math.cos(u[1]) * dt
    x[1] += u[0] * math.sin(u[1]) * dt
    x[2] = u[1]

    x[0] = np.minimum(x[0], 2428)
    x[1] = np.minimum(x[1], 2428)

    

    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob, global_path):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)
            to_global_path_cost = config.globalPath_cost_gain * calc_globalPath_cost(trajectory, global_path)

            final_cost = to_goal_cost + speed_cost + ob_cost + to_global_path_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK

def calc_globalPath_cost(trajectory, global_path):
    traj_end_xy = trajectory[-1][:2]
    min_dist_from_traj_end_to_globalPath = np.min(np.linalg.norm(global_path - traj_end_xy, axis=-1))
    return min_dist_from_traj_end_to_globalPath

def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    print(__file__ + " start!!")
    # INFO: generate a straight line as global path first
    global_traj_div = 1000
    global_path = np.stack((np.linspace(0, gx, global_traj_div), np.linspace(0, gx, global_traj_div)), axis=1)
    # INFO: initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # INFO: goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # input [forward speed, yaw_rate]

    config.robot_type = robot_type
    trajectory = np.array(x)
    ob = config.ob
    while True:
        u, predicted_trajectory = dwa_control(x, config, goal, ob, global_path)
        x = motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main(robot_type=RobotType.rectangle)
    # main(robot_type=RobotType.circle)
