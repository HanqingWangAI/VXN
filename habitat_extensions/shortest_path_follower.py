# Modified from https://github.com/facebookresearch/habitat-lab/blob/v0.1.4/habitat/tasks/nav/shortest_path_follower.py
# Use the Habitat v0.1.4 ShortestPathFollower for compatibility with
# the dataset generation oracle.

from typing import Optional, Union

import habitat_sim
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_two_vectors,
)
from habitat_extensions.utils import rvector_to_global_coordinates, rtheta_to_global_coordinates

from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
import math
from copy import deepcopy

EPSILON = 1e-6


def action_to_one_hot(action: int) -> np.array:
    one_hot = np.zeros(len(HabitatSimActions), dtype=np.float32)
    one_hot[action] = 1
    return one_hot


class ShortestPathFollowerCompat:
    """Utility class for extracting the action on the shortest path to the
        goal.
    Args:
        sim: HabitatSim instance.
        goal_radius: Distance between the agent and the goal for it to be
            considered successful.
        return_one_hot: If true, returns a one-hot encoding of the action
            (useful for training ML agents). If false, returns the
            SimulatorAction.
    """

    def __init__(
        self, sim: HabitatSim, goal_radius: float, return_one_hot: bool = True
    ):
        assert (
            getattr(sim, "geodesic_distance", None) is not None
        ), "{} must have a method called geodesic_distance".format(
            type(sim).__name__
        )

        self._sim = sim
        self._max_delta = sim.habitat_config.FORWARD_STEP_SIZE - EPSILON
        self._goal_radius = goal_radius
        self._step_size = sim.habitat_config.FORWARD_STEP_SIZE

        self._mode = (
            "geodesic_path"
            if getattr(sim, "get_straight_shortest_path_points", None)
            is not None
            else "greedy"
        )
        self._return_one_hot = return_one_hot

    def _get_return_value(self, action) -> Union[int, np.array]:
        if self._return_one_hot:
            return action_to_one_hot(action)
        else:
            return action

    def get_next_action(
        self, goal_pos: np.array
    ) -> Optional[Union[int, np.array]]:
        """Returns the next action along the shortest path."""
        if (
            self._sim.geodesic_distance(
                self._sim.get_agent_state().position, goal_pos
            )
            <= self._goal_radius
        ):
            return None

        max_grad_dir = self._est_max_grad_dir(goal_pos)
        if max_grad_dir is None:
            return self._get_return_value(HabitatSimActions.MOVE_FORWARD)
        return self._step_along_grad(max_grad_dir)

    def _step_along_grad(
        self, grad_dir: np.quaternion
    ) -> Union[int, np.array]:
        current_state = self._sim.get_agent_state()
        alpha = angle_between_quaternions(grad_dir, current_state.rotation)
        if alpha <= np.deg2rad(self._sim.habitat_config.TURN_ANGLE) + EPSILON:
            return self._get_return_value(HabitatSimActions.MOVE_FORWARD)
        else:
            sim_action = HabitatSimActions.TURN_LEFT
            self._sim.step(sim_action)
            best_turn = (
                HabitatSimActions.TURN_LEFT
                if (
                    angle_between_quaternions(
                        grad_dir, self._sim.get_agent_state().rotation
                    )
                    < alpha
                )
                else HabitatSimActions.TURN_RIGHT
            )
            self._reset_agent_state(current_state)
            return self._get_return_value(best_turn)

    def _reset_agent_state(self, state: habitat_sim.AgentState) -> None:
        self._sim.set_agent_state(
            state.position, state.rotation, reset_sensors=False
        )

    def _geo_dist(self, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(
            self._sim.get_agent_state().position, goal_pos
        )

    def _est_max_grad_dir(self, goal_pos: np.array) -> np.array:

        current_state = self._sim.get_agent_state()
        current_pos = current_state.position

        if self.mode == "geodesic_path":
            points = self._sim.get_straight_shortest_path_points(
                self._sim.get_agent_state().position, goal_pos
            )
            # Add a little offset as things get weird if
            # points[1] - points[0] is anti-parallel with forward
            if len(points) < 2:
                return None
            max_grad_dir = quaternion_from_two_vectors(
                self._sim.forward_vector,
                points[1]
                - points[0]
                + EPSILON
                * np.cross(self._sim.up_vector, self._sim.forward_vector),
            )
            max_grad_dir.x = 0
            max_grad_dir = np.normalized(max_grad_dir)
        else:
            current_rotation = self._sim.get_agent_state().rotation
            current_dist = self._geo_dist(goal_pos)

            best_geodesic_delta = -2 * self._max_delta
            best_rotation = current_rotation
            for _ in range(0, 360, self._sim.habitat_config.TURN_ANGLE):
                sim_action = HabitatSimActions.MOVE_FORWARD
                self._sim.step(sim_action)
                new_delta = current_dist - self._geo_dist(goal_pos)

                if new_delta > best_geodesic_delta:
                    best_rotation = self._sim.get_agent_state().rotation
                    best_geodesic_delta = new_delta

                # If the best delta is within (1 - cos(TURN_ANGLE))% of the
                # best delta (the step size), then we almost certainly have
                # found the max grad dir and should just exit
                if np.isclose(
                    best_geodesic_delta,
                    self._max_delta,
                    rtol=1
                    - np.cos(np.deg2rad(self._sim.habitat_config.TURN_ANGLE)),
                ):
                    break

                self._sim.set_agent_state(
                    current_pos,
                    self._sim.get_agent_state().rotation,
                    reset_sensors=False,
                )

                sim_action = HabitatSimActions.TURN_LEFT
                self._sim.step(sim_action)

            self._reset_agent_state(current_state)

            max_grad_dir = best_rotation

        return max_grad_dir

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode: str):
        """Sets the mode for how the greedy follower determines the best next
            step.
        Args:
            new_mode: geodesic_path indicates using the simulator's shortest
                path algorithm to find points on the map to navigate between.
                greedy indicates trying to move forward at all possible
                orientations and selecting the one which reduces the geodesic
                distance the most.
        """
        assert new_mode in {"geodesic_path", "greedy"}
        if new_mode == "geodesic_path":
            assert (
                getattr(self._sim, "get_straight_shortest_path_points", None)
                is not None
            )
        self._mode = new_mode


class ShortestPathFollowerOrientation:
    r"""Utility class for extracting the action on the shortest path to the
        goal.
    Args:
        sim: HabitatSim instance.
        goal_radius: Distance between the agent and the goal for it to be
            considered successful.
        return_one_hot: If true, returns a one-hot encoding of the action
            (useful for training ML agents). If false, returns the
            SimulatorAction.
    """

    def __init__(
        self, sim: HabitatSim, goal_radius: float, num_cameras: float, return_one_hot: bool = True
    ):
        assert (
            getattr(sim, "geodesic_distance", None) is not None
        ), "{} must have a method called geodesic_distance".format(type(sim).__name__)
        # print('enter the shortest path follower compat')
        # print(sim.config)
        # print(sim.habitat_config)

        self._sim = sim
        self._max_delta = self._sim.habitat_config.FORWARD_STEP_SIZE - EPSILON
        self._goal_radius = goal_radius
        self._step_size = self._sim.habitat_config.FORWARD_STEP_SIZE
        self._num_cameras = num_cameras

        self._mode = (
            "geodesic_path"
            if getattr(sim, "get_straight_shortest_path_points", None) is not None
            else "greedy"
        )
        self._return_one_hot = return_one_hot

    def _get_return_value(self, action) -> Union[int, np.array]:
        if self._return_one_hot:
            return action_to_one_hot(action)
        else:
            return action

    def get_next_orientation(self, goal_pos: np.array) -> Optional[Union[int, np.array]]:
        """Returns the next action along the shortest path.
        """
        # print('current',self._sim.get_agent_state().position,'goal', goal_pos)
        dis = self._sim.geodesic_distance(self._sim.get_agent_state().position, goal_pos[0])
        if (
            dis
            <= self._goal_radius
        ):
            return 0 # the heading is forward
        
        for goal in goal_pos:
            max_grad_dir = self._est_max_grad_dir(goal)
            if max_grad_dir is not None:
                break
        # max_grad_dir = self._est_max_grad_dir(goal_pos)
        if max_grad_dir is None:
            # print('no gradient, distance', self._sim.geodesic_distance(self._sim.get_agent_state().position, goal_pos))
            # return self._get_return_value(HabitatSimActions.STOP)
            return 0 # the heading is forward
        # return self._step_along_grad(max_grad_dir, dis, goal_pos[0])
        return self._get_relative_orientation(max_grad_dir)

    def _get_relative_orientation(self, grad_dir):
        num_dirs = self._num_cameras
        turn_angle = math.pi * 2 / num_dirs
        current_state = self._sim.get_agent_state()
        angle, w1 = quat_to_angle_axis(current_state.rotation)
        angle_grad, w2  = quat_to_angle_axis(grad_dir)

        sign1 = -1 if w1.sum() < 0 else 1
        sign2 = -1 if w2.sum() < 0 else 1
        
        alpha = (angle_grad * sign2 - angle * sign1)

        while alpha < 0:
            alpha += math.pi * 2
        while alpha > math.pi * 2:
            alpha -= math.pi * 2

        best_dir = int(alpha // turn_angle)
        
        # assert dis > new_dis, 'pre dis {} is smaller than current dis {}'.format(dis,new_dis)
        return best_dir


    def _step_along_grad(self, grad_dir: np.quaternion, cur_dis, goal) -> Union[int, np.array]:
        num_dirs = 360 // int(self._turn_angle) + (1 if (360 % self._turn_angle != 0) else 0)
        current_state = deepcopy(self._sim.get_agent_state())

        angle, w1 = quat_to_angle_axis(current_state.rotation)
        angle_grad, w2  = quat_to_angle_axis(grad_dir)

        self._sim.set_agent_state(current_state.position, grad_dir)
        self._sim.step(HabitatSimActions.MOVE_FORWARD)
        next_state = self._sim.get_agent_state()
        next_dis = self._sim.geodesic_distance(next_state.position, goal)
        flag = next_dis >= cur_dis
        if flag:
            # if angle < 0 or angle_grad < 0:
            #     print('< 0', 'angle',angle, 'angle_grad',angle_grad)
            sign = -1 if w1.sum() < 0 else 1
            # cands = [-1,0,1]
            cands = range(num_dirs)
            sign2 = -1 if w2.sum() < 0 else 1
            # print('grad angle', angle_grad * sign2, 'current angle',angle*sign)
            best_angle = None
            min_dis = 1e10
            for i in cands:
                angle_cand = angle_grad + np.deg2rad(self._turn_angle) * sign2 * i
                while angle_cand < 0:
                    angle_cand += math.pi * 2
                while angle_cand > math.pi * 2:
                    angle_cand -= math.pi * 2
                rot = quat_from_angle_axis(angle_cand, np.array([0,1 * sign2,0]))
                self._sim.set_agent_state(current_state.position, rot)
                self._sim.step(HabitatSimActions.MOVE_FORWARD)
                next_state = self._sim.get_agent_state()
                next_dis = self._sim.geodesic_distance(next_state.position, goal)
                print(i, 'next_dis', next_dis, 'cur_dis', cur_dis,'turned angle', angle_cand)
                if next_dis < min_dis:
                    best_angle = angle_cand
                    min_dis = next_dis

            grad_dir = quat_from_angle_axis(best_angle, np.array([0,1 * sign2,0]))

        alpha = angle_between_quaternions(grad_dir, current_state.rotation)

        self._reset_agent_state(current_state)
        # print('angle',alpha * 180 / np.math.pi, 'rotation', current_state.rotation)
        

     
        # print('angleangle', angle, w)
        if alpha <= np.deg2rad(self._sim.habitat_config.TURN_ANGLE) + EPSILON:
            # return self._get_return_value(HabitatSimActions.MOVE_FORWARD)
            # if flag:
            #     angle, w1 = quat_to_angle_axis(current_state.rotation)

            #     sign = -1 if w1.sum() < 0 else 1
            #     angle_after_turn = angle

            #     while angle_after_turn < 0:
            #         angle_after_turn += math.pi * 2
            #     while angle_after_turn > math.pi * 2:
            #         angle_after_turn -= math.pi * 2

            #     rotation = quat_from_angle_axis(angle_after_turn, np.array([0,1 * sign,0]))

            #     self._sim.set_agent_state(current_state.position, rotation, reset_sensors=False)

            #     self._sim.step(HabitatSimActions.MOVE_FORWARD)
            #     next_state = self._sim.get_agent_state()
            #     next_dis = self._sim.geodesic_distance(next_state.position, goal)

            #     self._reset_agent_state(current_state)

                # print('best angle', best_angle, 'angle_after_turn', angle_after_turn,'next_dis',next_dis, 'cur_dis',cur_dis)
            return 1 # move forward
        else:
            # sim_action = HabitatSimActions.TURN_LEFT
            # self._sim.step(sim_action)
            sign = -1 if w1.sum() < 0 else 1
            angle_after_turn = angle + np.deg2rad(self._turn_angle) * sign
            rotation = quat_from_angle_axis(angle_after_turn, np.array([0,1 * sign,0]))
            # print('after turn', rotation)
            beta = angle_between_quaternions(
                        grad_dir, rotation
                    )
            # print('angle',angle, 'after turn', beta, 'before', alpha, 'grad_angle', angle_grad, 'w1',w1,'w2',w2)
            best_turn = (
                HabitatSimActions.TURN_LEFT
                if (
                    beta
                    < alpha
                )
                else HabitatSimActions.TURN_RIGHT
            )
            
            if best_turn == HabitatSimActions.TURN_LEFT:
                best_action = num_dirs - int(round(alpha/np.deg2rad(self._turn_angle))) + 1
                # if int(round(alpha/np.deg2rad(self._turn_angle))) * np.deg2rad(self._turn_angle) > alpha:
                #     second_best = ((best_action - 2 + num_dirs) % num_dirs) + 1
                # else:
                #     second_best = (best_action % num_dirs)  + 1
            else:
                best_action = int(round(alpha/np.deg2rad(self._turn_angle))) + 1
                # if int(round(alpha/np.deg2rad(self._turn_angle))) * np.deg2rad(self._turn_angle) > alpha:
                #     second_best = (best_action % num_dirs)  + 1
                # else:
                #     second_best = ((best_action - 2 + num_dirs) % num_dirs) + 1

            # self._sim.set_agent_state(current_state.position, grad_dir)
            # self._sim.step(HabitatSimActions.MOVE_FORWARD)
            # next_state = self._sim.get_agent_state()
            # next_dis = self._sim.geodesic_distance(next_state.position, goal)
            # if next_dis >= cur_dis:
            #     best_action = 0

            # print('best turn is', 'left' if best_turn == HabitatSimActions.TURN_LEFT else 'right', best_action)
            # print('best turn',best_turn)
            # if flag:
            #     angle, w1 = quat_to_angle_axis(current_state.rotation)

            #     sign = -1 if w1.sum() < 0 else 1
            #     angle_after_turn = angle - np.deg2rad(self._turn_angle) * sign * (best_action - 1) # turing right

            #     while angle_after_turn < 0:
            #         angle_after_turn += math.pi * 2
            #     while angle_after_turn > math.pi * 2:
            #         angle_after_turn -= math.pi * 2

            #     rotation = quat_from_angle_axis(angle_after_turn, np.array([0,1 * sign,0]))

            #     self._sim.set_agent_state(current_state.position, rotation, reset_sensors=False)

            #     self._sim.step(HabitatSimActions.MOVE_FORWARD)
            #     next_state = self._sim.get_agent_state()
            #     next_dis = self._sim.geodesic_distance(next_state.position, goal)

                # print('best angle', best_angle, 'angle_after_turn', angle_after_turn,'next_dis',next_dis, 'cur_dis',cur_dis)

            # self._reset_agent_state(current_state)
            
            # return self._get_return_value(best_turn)
            assert best_action <= num_dirs and best_action > 0, 'best action error, is %d'%best_action
            return best_action

    def _reset_agent_state(self, state: habitat_sim.AgentState) -> None:
        self._sim.set_agent_state(state.position, state.rotation, reset_sensors=False)

    def _geo_dist(self, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(
            self._sim.get_agent_state().position, goal_pos
        )

    def _est_max_grad_dir(self, goal_pos: np.array) -> np.array:

        current_state = self._sim.get_agent_state()
        current_pos = current_state.position
        # print('mode',self.mode, 'forward vector', self._sim.forward_vector)
        if self.mode == "geodesic_path":
            points = self._sim.get_straight_shortest_path_points(
                self._sim.get_agent_state().position, goal_pos
            )
            # print('points', points)
            # Add a little offset as things get weird if
            # points[1] - points[0] is anti-parallel with forward
            if len(points) < 2:
                return None
            points = np.array(points)
            # print(points[1], points[0])
            max_grad_dir = quaternion_from_two_vectors(
                self._sim.forward_vector,
                points[1]
                - points[0]
                + EPSILON * np.cross(self._sim.up_vector, self._sim.forward_vector),
            )
            # print('forward vector', self._sim.forward_vector)
            # print('gradient', points[1]
            #     - points[0]
            #     + EPSILON * np.cross(self._sim.up_vector, self._sim.forward_vector))
            max_grad_dir.x = 0
            max_grad_dir = np.normalized(max_grad_dir)
            # print('norm grad', max_grad_dir)
        else:
            current_rotation = self._sim.get_agent_state().rotation
            current_dist = self._geo_dist(goal_pos)
            # print('current_dist', current_dist)
            best_geodesic_delta = -2 * self._max_delta
            best_rotation = current_rotation
            for _ in range(0, 360, self._sim.habitat_config.TURN_ANGLE):
                sim_action = HabitatSimActions.MOVE_FORWARD
                self._sim.step(sim_action)
                new_delta = current_dist - self._geo_dist(goal_pos)

                if new_delta > best_geodesic_delta:
                    best_rotation = self._sim.get_agent_state().rotation
                    best_geodesic_delta = new_delta

                # If the best delta is within (1 - cos(TURN_ANGLE))% of the
                # best delta (the step size), then we almost certainly have
                # found the max grad dir and should just exit
                if np.isclose(
                    best_geodesic_delta,
                    self._max_delta,
                    rtol=1 - np.cos(np.deg2rad(self._sim.habitat_config.TURN_ANGLE)),
                ):
                    break

                self._sim.set_agent_state(
                    current_pos,
                    self._sim.get_agent_state().rotation,
                    reset_sensors=False,
                )

                sim_action = HabitatSimActions.TURN_LEFT
                self._sim.step(sim_action)

            self._reset_agent_state(current_state)

            max_grad_dir = best_rotation

        return max_grad_dir

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode: str):
        r"""Sets the mode for how the greedy follower determines the best next
            step.
        Args:
            new_mode: geodesic_path indicates using the simulator's shortest
                path algorithm to find points on the map to navigate between.
                greedy indicates trying to move forward at all possible
                orientations and selecting the one which reduces the geodesic
                distance the most.
        """
        assert new_mode in {"geodesic_path", "greedy"}
        if new_mode == "geodesic_path":
            assert (
                getattr(self._sim, "get_straight_shortest_path_points", None)
                is not None
            )
        self._mode = new_mode


class ShortestPathFollowerWaypoint(ShortestPathFollowerCompat):
    
    def _est_max_grad_dir_bk(self, goal_pos: np.array) -> np.array:

        current_state = self._sim.get_agent_state()
        current_pos = current_state.position

        if self.mode == "geodesic_path":
            points = self._sim.get_straight_shortest_path_points(
                self._sim.get_agent_state().position, goal_pos
            )

            ori_dis = self._sim.geodesic_distance(
                    self._sim.get_agent_state().position, goal_pos
                )

            with open('data/logs.txt','a') as fp:
                fp.write(f'ori dis {ori_dis}\n')
            # Add a little offset as things get weird if
            # points[1] - points[0] is anti-parallel with forward
            if len(points) < 2:
                return None
            
            

            dis = 100
            max_grad_dir = None
            
            # print('ori dis', self._sim.geodesic_distance(
            #         self._sim.get_agent_state().position, goal_pos
            #     ))
            for p in points[1:]: # to find a best grad dir to avoid stuck
                grad_vector = p - points[0]
                grad_vector[1] = 0
                tmp_max_grad_dir = quaternion_from_two_vectors(
                    self._sim.forward_vector,
                    grad_vector
                    + EPSILON
                    * np.cross(self._sim.up_vector, self._sim.forward_vector),
                )
                
                tmp_max_grad_dir.x = 0
                tmp_max_grad_dir = np.normalized(tmp_max_grad_dir)
                # with open('data/logs.txt','a') as fp:
                #     vec = p-points[0]
                #     forw = self._sim.forward_vector
                #     lstr = f'forward vector, ({forw[0]},{forw[1]},{forw[2]}) grad vector ({vec[0]},{vec[1]},{vec[2]}),\n'
                #     fp.write(lstr)
                    

                theta = self._get_theta(tmp_max_grad_dir)
                # pos = rvector_to_global_coordinates(
                #     self._sim, self._step_size, tmp_max_grad_dir, y_delta=0, dimensionality=3
                # )
                pos = rtheta_to_global_coordinates(self._sim, self._step_size, theta, 0, 3)


                agent_pos = self._sim.get_agent_state().position
                new_pos = np.array(self._sim.step_filter(agent_pos, pos))
                
                new_dis = self._sim.geodesic_distance(
                    new_pos, goal_pos
                )
                # with open('data/logs.txt','a') as fp:
                #     fp.write(f'new dis {new_dis}\n')
                # print('new dis', new_dis)
                if new_dis > dis:
                    continue
                else:
                    dis = new_dis
                    max_grad_dir = tmp_max_grad_dir
                # break
                
        else:
            current_rotation = self._sim.get_agent_state().rotation
            current_dist = self._geo_dist(goal_pos)

            best_geodesic_delta = -2 * self._max_delta
            best_rotation = current_rotation
            for _ in range(0, 360, self._sim.habitat_config.TURN_ANGLE):
                sim_action = HabitatSimActions.MOVE_FORWARD
                self._sim.step(sim_action)
                new_delta = current_dist - self._geo_dist(goal_pos)

                if new_delta > best_geodesic_delta:
                    best_rotation = self._sim.get_agent_state().rotation
                    best_geodesic_delta = new_delta

                # If the best delta is within (1 - cos(TURN_ANGLE))% of the
                # best delta (the step size), then we almost certainly have
                # found the max grad dir and should just exit
                if np.isclose(
                    best_geodesic_delta,
                    self._max_delta,
                    rtol=1
                    - np.cos(np.deg2rad(self._sim.habitat_config.TURN_ANGLE)),
                ):
                    break

                self._sim.set_agent_state(
                    current_pos,
                    self._sim.get_agent_state().rotation,
                    reset_sensors=False,
                )

                sim_action = HabitatSimActions.TURN_LEFT
                self._sim.step(sim_action)

            self._reset_agent_state(current_state)

            max_grad_dir = best_rotation

        return max_grad_dir

    def _est_max_grad_dir(self, points):

        current_state = self._sim.get_agent_state()
        current_pos = current_state.position

        if len(points) < 2:
            return None
        # ori_dis = self._sim.geodesic_distance(
        #         current_pos, points[-1]
        #     )

        # with open('data/logs.txt','a') as fp:
        #     fp.write(f'ori dis {ori_dis}\n')
        dis = 100
        max_grad_dir = None

        
        
        # print('ori dis', self._sim.geodesic_distance(
        #         self._sim.get_agent_state().position, goal_pos
        #     ))
        for p in points[1:]: # to find a best grad dir to avoid stuck
            grad_vector = p - points[0]
            grad_vector[1] = 0
            tmp_max_grad_dir = quaternion_from_two_vectors(
                self._sim.forward_vector,
                grad_vector
                + EPSILON
                * np.cross(self._sim.up_vector, self._sim.forward_vector),
            )
            
            tmp_max_grad_dir.x = 0
            tmp_max_grad_dir = np.normalized(tmp_max_grad_dir)
            # with open('data/logs.txt','a') as fp:
            #     vec = p-points[0]
            #     forw = self._sim.forward_vector
            #     lstr = f'forward vector, ({forw[0]},{forw[1]},{forw[2]}) grad vector ({vec[0]},{vec[1]},{vec[2]}),\n'
            #     fp.write(lstr)
                
            max_grad_dir = tmp_max_grad_dir
            # theta = self._get_theta(tmp_max_grad_dir)
            # # pos = rvector_to_global_coordinates(
            # #     self._sim, self._step_size, tmp_max_grad_dir, y_delta=0, dimensionality=3
            # # )
            # pos = rtheta_to_global_coordinates(self._sim, self._step_size, theta, 0, 3)


            # agent_pos = self._sim.get_agent_state().position
            # new_pos = np.array(self._sim.step_filter(agent_pos, pos))
            
            # new_dis = self._sim.geodesic_distance(
            #     new_pos, points[-1]
            # )
            # with open('data/logs.txt','a') as fp:
            #     fp.write(f'new dis {new_dis}\n')
            # # print('new dis', new_dis)
            # if new_dis > dis:
            #     continue
            # else:
            #     dis = new_dis
            #     max_grad_dir = tmp_max_grad_dir
            break
                

        return max_grad_dir
         
    def get_next_action(self, goal_pos: np.array, episode=None) -> Optional[Union[int, np.array]]:
        """Returns the next action along the shortest path.
        """
        path = self._sim.get_shortest_path(
                    self._sim.get_agent_state().position, goal_pos, episode
                )
        dis = path.geodesic_distance
        points = path.points
        
        # print('dis', dis, 'radius', self._goal_radius)
        _num_cameras = 360 // self._sim.habitat_config.TURN_ANGLE
        if (
            dis
            <= self._goal_radius
        ):
            return {"action": "STOP"}, {"pano": _num_cameras, "offset": 0, "distance": 0,}
        
        max_grad_dir = self._est_max_grad_dir(points)
        # for goal in goal_pos:
        #     max_grad_dir = self._est_max_grad_dir(goal)
        #     if max_grad_dir is not None:
        #         break

        if max_grad_dir is None:
            return {"action": "STOP"}, {"pano": _num_cameras, "offset": 0, "distance": 0,}

        return self._step_along_grad(max_grad_dir)

    def _get_theta(self, grad_dir):
        current_state = self._sim.get_agent_state()
        alpha = angle_between_quaternions(grad_dir, current_state.rotation)
        # print(alpha, angle_between_quaternions( current_state.rotation, grad_dir))
        # print('angle',alpha * 180 / np.math.pi, 'rotation', current_state.rotation)
        # theta = -alpha
        angle, w1 = quat_to_angle_axis(current_state.rotation)

        sign = -1 if w1.sum() < 0 else 1
        angle_after_turn = angle + 0.05 * sign
        rotation = quat_from_angle_axis(angle_after_turn, np.array([0, 1 * sign,0]))
            # print('after turn', rotation)
        beta = angle_between_quaternions(
                    grad_dir, rotation
                )

        if beta < alpha:
            theta = alpha
        else:
            theta = -alpha

        theta = theta % (2 * math.pi)
        return theta

    def _step_along_grad(self, grad_dir: np.quaternion) -> Union[int, np.array]:
        _num_cameras = 360 // self._sim.habitat_config.TURN_ANGLE
        turn_angle = np.deg2rad(self._sim.habitat_config.TURN_ANGLE)
        theta = self._get_theta(grad_dir)

        pano_stop = ((theta + (0.5 * turn_angle)) // turn_angle) % _num_cameras
        offset = (theta % turn_angle) - 0.5 * turn_angle
        action_elements = {
            "pano": pano_stop,
            "offset": offset,
            "distance": self._step_size,
        }

        return {"action": {
                "action": "GO_TOWARD_POINT",
                "action_args": {
                                "r": self._step_size,
                                "theta": theta,
                            },

            }
        }, action_elements
