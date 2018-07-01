import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import math

import planning_utils as pu

#from planning_utils import a_star, heuristic, create_grid
#from planning_utils import a_star, heuristic, create_grid, prune_path, create_grid_and_edges, a_starGraph, GetGraph, closest_point, GetGridAndOffsets, GetLat0Lon0
import planning_utils as pu
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):
    
    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        self.goal_North = None
        self.goal_East = None

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < self.get_waypoint_transition_threshold():
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()
                        
    def get_waypoint_transition_threshold(self):
        result = 1.0
        
        if len(self.waypoints) > 2:        
            velocity = np.linalg.norm(self.local_velocity[0:2])        
            result = 2.5*math.exp(-1/np.clip(velocity, 1, 10))
        
        return result

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        if(self.waypoints):
            self.target_position = self.waypoints.pop(0)
        else:
            self.landing_transition()
            return
            
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")

        self.target_position[2] = pu.TARGET_ALTITUDE
        
        # TODO: read lat0, lon0 from colliders into floating point values
        lat0, lon0 = pu.GetLat0Lon0()
        
        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)

        # TODO: retrieve current global position
        global_position = self.global_position
 
        # TODO: convert to current local position using global_to_local()        
        local_position = global_to_local(global_position, self.global_home)
        
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))        
        
        grid, north_offset, east_offset = pu.GetGridAndOffsets()
        graph = pu.GetVoronoyGraph()
        
        # Define starting point on the grid (this is just grid center)
        # grid_start = (-north_offset, -east_offset)
        # TODO: convert start position to current position rather than map center
        start = (int(local_position[0]-north_offset), int(local_position[1])-east_offset)
        
        # Set goal as some arbitrary position on the grid        
        goal = (250,720)# (-north_offset + 10, -east_offset + 10)
                
        
        # TODO: adapt to set goal as latitude / longitude position and convert
        goal_position = global_to_local([-122.3962, 37.7950, 0.0], self.global_home)
        goal = (int(goal_position[0])-north_offset, int(goal_position[1])-east_offset)        
        
        if(not self.goal_North is None and not self.goal_East is None):
            goal = (self.goal_North, self.goal_East)

        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)        
        
        waypointsPath = self.GetPath(north_offset, east_offset, grid, graph, start, goal)
        if(waypointsPath is None):
            print("Path was not found")
            waypointsPath = []
              
        self.waypoints = waypointsPath
        print(self.waypoints)
        self.send_waypoints()

    def GetPath(self, north_offset, east_offset, grid, graph, start, goal):
        
        if(grid[goal[0], goal[1]] == 1):
            print("Goal is unreachable")
            return None
        
        start_g = pu.closest_point(graph, start)
        goal_g = pu.closest_point(graph, goal)
        print(goal_g)
        
        path, _ = pu.a_starGraph(graph, start_g, goal_g)
        # If it is not possible to find a path in tha graph, let's find a path in the grid
        if(path is None):
            print('Searching in grid')
            newpath, _ = pu.a_star(grid, start, goal)
            newpath = pu.prune_path(newpath)
        else:            
            start_g = (int(start_g[0]), int(start_g[1]))
            goal_g = (int(goal_g[0]), int(goal_g[1]))
        
            print('start={0}, start_g={1}'.format(start, start_g))
            pathToBegin, _ = pu.a_star(grid, start, start_g)
            pathFromEnd, _ = pu.a_star(grid, goal_g, goal)
        
            if(pathToBegin is None or pathFromEnd is None):
                return None
        
            newpath = pathToBegin + path + pathFromEnd
            newpath = pu.prune_path(newpath)
        
        return [[int(p[0]) + north_offset, int(p[1]) + east_offset, pu.TARGET_ALTITUDE, 0] for p in newpath]
    
    def start(self, goal_North=None, goal_East=None):
        self.start_log("Logs", "NavLog.txt")

        self.goal_North = goal_North
        self.goal_East = goal_East

        print("starting connection")
        self.connection.start()
        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    parser.add_argument('--n', type=int, default=None, help='goal north position')
    parser.add_argument('--e', type=int, default=None, help='goal east position')
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start(args.n, args.e)
