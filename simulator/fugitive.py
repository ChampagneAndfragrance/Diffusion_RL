from simulator.abstract_object import MovingObject, DetectionObject
from simulator.terrain import TerrainType
import simulator.helicopter
import simulator.search_party
import simulator.camera
import numpy as np


class Fugitive(DetectionObject, MovingObject):
    def __init__(self, terrain, location, fugitive_speed_limit):
        """
        Fugitive defines the fugitive. Initializes detection parameters.
        :param terrain: a terrain instance
        :param location: a list of length 2. For example, [5, 7]
        """
        DetectionObject.__init__(self, terrain, location, detection_object_type_coefficient=0.5)
        MovingObject.__init__(self, terrain, location, speed=fugitive_speed_limit)
        # NOTE: the detection_object_type_coefficient is variant for fugitive as it is detecting different objects

    def detect(self, location_object, object_instance, timestep, prisoner_loc=None):
        """
        Determine detection of an object based on its location and type of the object
        The fugitive's detection of other parties is different other parties' detection of the fugitive as given in the "detection ranges.xlsx".
        The fugitive's detection of other parties depends on what the party is.
        :param location_object:
        :param object_instance: the instance referred to the object the fugitive is detecting.
        :return: [b,x,y] where b is a boolean indicating detection, and [x,y] is the location of the object in world coordinates if b=True, [x,y]=[-1,-1] if b=False
        """
        if isinstance(object_instance, simulator.helicopter.Helicopter):
            self.detection_object_type_coefficient = 0.7 # 0.5
            return DetectionObject.detect(self, location_object, 127, timestep, 0, prisoner_loc) # 24, 127
            # return DetectionObject.detect(self, location_object, np.linalg.norm(object_instance.step_dist_xy), timestep, 0, prisoner_loc) # 24, 127
        elif isinstance(object_instance, simulator.search_party.SearchParty):
            self.detection_object_type_coefficient = 3.5 # 0.75, 5.0
            return DetectionObject.detect(self, location_object, 20, timestep, 1, prisoner_loc) # 24, 10
            # return DetectionObject.detect(self, location_object, np.linalg.norm(object_instance.step_dist_xy), timestep, 1, prisoner_loc) # 24, 10
        elif isinstance(object_instance, simulator.camera.Camera):
            self.detection_object_type_coefficient = 10.0 # 1.0, 3.0
            return DetectionObject.detect(self, location_object, 4, timestep, 2, prisoner_loc) # 8
        else:
            raise NotImplementedError
