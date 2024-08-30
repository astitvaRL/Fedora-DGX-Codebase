import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.cm as cm
from scipy.optimize import minimize
from typing import Tuple
import time
from matplotlib.animation import FuncAnimation
from animated_drawings.model.vectors import Vectors


class ViewCostMinimizer():
    def __init__(self):
        self.great_circle_lat1 = None
        self.great_circle_lat2 = None
        self.great_circle_lon1 = None
        self.great_circle_lon2 = None
        self.great_circle_cost_weighting = None  # modify cost weighting of great circle depending upon how straight the limb is

        self.global_viewing_angle_lat = None
        self.global_viewing_angle_lon = None

        self.last_proj_lat = None
        self.last_proj_lon = None

    def set_bone_vectors(self, bone_vector1_xyz, bone_vector2_xyz):
        self.great_circle_lat1, self.great_circle_lon1 = cartesian_to_lat_lon(*bone_vector1_xyz)
        self.great_circle_lat2, self.great_circle_lon2 = cartesian_to_lat_lon(*bone_vector2_xyz)
        bv1 = Vectors(bone_vector1_xyz)
        bv1.norm()
        bv2 = Vectors(bone_vector2_xyz)
        bv2.norm()
        self.great_circle_cost_weighting = 1 - np.dot(bv1.vs[0], bv2.vs[0])  # when dot product is 1, limbs at straight, so great circle weighting cost should be zero


    def set_global_view_vector(self, global_viewing_vector_xyz):
        self.global_viewing_angle_lat, self.global_viewing_angle_lon = cartesian_to_lat_lon(*global_viewing_vector_xyz)

    def set_last_view_vector(self, last_projection_vector):
        self.last_proj_lat, self.last_proj_lon = cartesian_to_lat_lon(*last_projection_vector)

    def compute_cost_from_lat_lon(self, lat_lon):
        lat, lon = lat_lon
        distance_to_great_circle = distance_to_great_circle_chatgpt((lat, lon), (self.great_circle_lat1, self.great_circle_lon1), (self.great_circle_lat2, self.great_circle_lon2))
        distance_to_viewing_angle = great_circle_distance((lat, lon), (self.global_viewing_angle_lat, self.global_viewing_angle_lon))
        distance_to_last_projection_vector = great_circle_distance((lat, lon), (self.last_proj_lat, self.last_proj_lon))

        # TODO: Add these to a config file somehow
        gc_cost = 2 * self.great_circle_cost_weighting
        return modified_gaussian(distance_to_great_circle, gc_cost, 0.3, 1) + \
            modified_gaussian(distance_to_viewing_angle, -2.5, 0.5, 0) + \
            modified_gaussian(distance_to_last_projection_vector, -1.0, 0.3, 0)

    def minimize(self):

        if None in [self.great_circle_lat1, self.great_circle_lat2, self.great_circle_lon1, self.great_circle_lon2, self.global_viewing_angle_lat, self.global_viewing_angle_lon, self.last_proj_lat, self.last_proj_lon]:
            assert False, 'not properly initialized'

        answer1 = minimize(self.compute_cost_from_lat_lon, np.array([self.global_viewing_angle_lat, self.global_viewing_angle_lon]), tol=0.003, method='BFGS')
        answer2 = minimize(self.compute_cost_from_lat_lon, np.array([self.last_proj_lat, self.last_proj_lon]), tol=0.003, method='BFGS')

        if math.isnan(answer1.fun) and math.isnan(answer2.fun):
            print('both are nan')
            answer = answer1
        elif math.isnan(answer1.fun):
            answer = answer2
        elif math.isnan(answer2.fun):
            answer = answer1
        else:
            if answer1.fun < answer2.fun:
                answer = answer1
            else:
                answer = answer2

        answer_xyz = lat_lon_to_cartesian(*answer.x)

        return answer_xyz
    
    def visualize_unit_sphere(self):
        answer_xyz = self.minimize()

        num_points = 80
        latitudes = np.linspace(-90, 90, num_points)
        longitudes = np.linspace(-180, 180, num_points)

        # Generate the Cartesian coordinates and colors for each point on the sphere
        coords = []
        colors = []
        distances = []
        for lat in latitudes:
            for lon in longitudes:
                coords.append(lat_lon_to_cartesian(lat, lon))
                distances.append(self.compute_cost_from_lat_lon(np.array([lat, lon])))

        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        for distance in distances:
            colors.append(cm.jet(distance))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('Unit Sphere')

        ax.scatter(*zip(*coords), s=5, c=colors, alpha=0.7)
        ax.scatter(*(1.1*np.array(answer_xyz)).tolist(), color='red', s=20, marker='x')
        ax.text(*(1.1*np.array(answer_xyz)).tolist(), "Minimum", color='black', zorder=2)

        # add the global viewing vector
        # viewing_vector_xyz = lat_lon_to_cartesian(*cartesian_to_lat_lon(*viewing_vector))
        viewing_vector_xyz = lat_lon_to_cartesian(self.global_viewing_angle_lat, self.global_viewing_angle_lon)
        ax.scatter(*(1.1*np.array(viewing_vector_xyz)).tolist(), color='green', s=20, marker='x')

        # add the last frame view vector
        # last_projection_vector_xyz = lat_lon_to_cartesian(*cartesian_to_lat_lon(*last_projection_vector))
        last_projection_vector_xyz = lat_lon_to_cartesian(self.last_proj_lat, self.last_proj_lon)
        ax.scatter(*(1.1*np.array(last_projection_vector_xyz)).tolist(), color='yellow', s=20, marker='x')

        plt.show()


    def reset(self):
        self.great_circle_lat1 = None
        self.great_circle_lat2 = None
        self.great_circle_lon1 = None
        self.great_circle_lon2 = None

        self.global_viewing_angle_lat = None
        self.global_viewing_angle_lon = None

        self.last_proj_lat = None
        self.last_proj_lon = None

def lat_lon_to_cartesian(lat, lon):
    """
    Convert latitude and longitude to Cartesian coordinates on a unit sphere.

    Parameters:
    lat (float): Latitude in degrees.
    lon (float): Longitude in degrees.

    Returns:
    tuple: Cartesian coordinates (x, y, z).
    """
    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate Cartesian coordinates
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return x, y, z


def cartesian_to_lat_lon(x: float, y: float, z: float) -> Tuple[float, float]:
    """
    Convert Cartesian coordinates to latitude and longitude on a unit sphere.
    If the point defined by the Cartesian coordinates does not lie on the unit sphere,
    a warning message is printed and the vector is normalized.
    Parameters:
    x (float): X coordinate.
    y (float): Y coordinate.
    z (float): Z coordinate.
    Returns:
    tuple: Latitude and longitude in degrees.
    """
    # Calculate the distance from the origin
    distance: float = np.sqrt(x**2 + y**2 + z**2)
    # Check if the point lies on the unit sphere
    if not np.isclose(distance, 1):
        #print("Warning: The point does not lie on the unit sphere. Normalizing the vector.")
        x /= distance
        y /= distance
        z /= distance
    # Calculate latitude and longitude
    lat: float = np.arcsin(z)  # Latitude
    lon: float = np.arctan2(y, x)  # Longitude
    # Convert latitude and longitude from radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)
    return lat, lon

def lat_lon_to_distance(lat_lon: npt.NDArray[np.float]):
    """
    Convert latitude and longitude to a color.
    Parameters:
    lat (float): Latitude in degrees.
    lon (float): Longitude in degrees.
    Returns:
    tuple: RGB color.
    """
    # global viewing_vector
    # global last_projection_vector
    # global bone_vector1
    # global bone_vector2

    # # Normalize latitude and longitude to [0, 1]
    # lat_norm = (lat + 90) / 180
    # lon_norm = (lon + 180) / 360
    # # Map normalized latitude and longitude to a color using a colormap
    # return cm.jet(lat_norm)  #  * lon_norm)

    # return cm.jet(great_circle_distance((lat, lon), (0, 0)) / np.pi)

    #return cm.jet(great_circle_distance_to_arc_chatgpt((lat, lon), (5, 5), (15, 95)) / np.pi)

    global great_circle_lat1
    global great_circle_lat2
    global great_circle_lon1
    global great_circle_lon2

    global global_viewing_angle_lat
    global global_viewing_angle_lon

    global last_proj_lat
    global last_proj_lon


    lat, lon = lat_lon
    distance_to_great_circle = distance_to_great_circle_chatgpt((lat, lon), (great_circle_lat1, great_circle_lon1), (great_circle_lat2, great_circle_lon2))
    distance_to_viewing_angle = great_circle_distance((lat, lon), (global_viewing_angle_lat, global_viewing_angle_lon))
    distance_to_last_projection_vector = great_circle_distance((lat, lon), (last_proj_lat, last_proj_lon))

    return modified_gaussian(distance_to_great_circle, 2, 0.3, 1) + \
        modified_gaussian(distance_to_viewing_angle, -1.5, 0.5, 0) # + \
        # modified_gaussian(distance_to_last_projection_vector, -0.0, 0.5, 0)


def modified_gaussian(distance, A, sigma, C):
    """
    Modified Gaussian function that asymptotically approaches a value C.

    Args:
    x (float or array): Input value(s).
    A (float): Amplitude of the peak.
    x0 (float): Point where the function asymptotically approaches.
    sigma (float): Controls the rate of fall-off.
    C (float): Asymptotic value as x moves away from x0.
    
    distance - distance to the great circle

    Returns:
    float or array: Output value of the function.
    """
    return A * np.exp(-((distance)**2) / (2 * sigma**2)) + C

def great_circle_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the great circle distance between two points on a unit sphere.
    Parameters:
    point1 (tuple): The latitude and longitude of the first point in degrees.
    point2 (tuple): The latitude and longitude of the second point in degrees.
    Returns:
    float: The great circle distance between the two points.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(point1)
    lat2, lon2 = np.radians(point2)
    # Calculate the differences
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    # Calculate the great circle distance using the spherical law of cosines
    great_circle_distance = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(delta_lon))
    return great_circle_distance

def distance_to_great_circle_chatgpt(point: Tuple[float, float], gc_point1: Tuple[float, float], gc_point2: Tuple[float, float]) -> float:


    # TODO: replace with cross-track distance from https://www.movable-type.co.uk/scripts/latlong.html


    """
    Calculate the shortest distance from a point to a great circle defined by two points on a sphere.

    Args:
    point (Tuple[float, float]): The latitude and longitude of the point (in degrees).
    gc_point1 (Tuple[float, float]): The latitude and longitude of the first point on the great circle (in degrees).
    gc_point2 (Tuple[float, float]): The latitude and longitude of the second point on the great circle (in degrees).

    Returns:
    float: The shortest distance from the point to the great circle on the surface of the sphere (in the same units as the sphere's radius).

    The function assumes a unit sphere. If working with Earth or another sphere, multiply the result by the sphere's radius to get the actual distance.
    """

    # Helper function to convert degrees to radians
    def to_radians(degrees: float) -> float:
        return degrees * math.pi / 180

    # Convert coordinates from degrees to radians
    lat1, lon1 = map(to_radians, point)
    lat2, lon2 = map(to_radians, gc_point1)
    lat3, lon3 = map(to_radians, gc_point2)

    # Convert spherical coordinates to Cartesian coordinates
    def to_cartesian(lat, lon):
        return [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]

    # Cartesian coordinates for the point and the great circle points
    p = to_cartesian(lat1, lon1)
    a = to_cartesian(lat2, lon2)
    b = to_cartesian(lat3, lon3)

    # Calculate the normal vector to the plane of the great circle
    def cross_product(v1, v2):
        return [v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]]

    normal = cross_product(a, b)

    # Calculate the dot product for the projection
    def dot_product(v1, v2):
        return sum(x*y for x, y in zip(v1, v2))

    # Calculate the distance from the point to the great circle
    # This is the angle between the point vector and its projection onto the great circle plane
    projection_length = dot_product(p, normal) / math.sqrt(dot_product(normal, normal))
    angle = math.asin(min(1, max(-1, projection_length)))  # Clamp the value to avoid numerical errors

    return abs(angle)


if __name__ == '__main__':
    frames = []

    view_cost_minizer = ViewCostMinimizer()
    last_projection_vector = None
    for idx in range(-10, 10):

        # set up the vectors
        bone_vector1 = (0, -1, 1)
        bone_vector2 = (0, -1, -1)
        viewing_vector = (idx/10, 0.0, 1.0)

        view_cost_minizer.set_bone_vectors(bone_vector1, bone_vector2)
        view_cost_minizer.set_global_view_vector(viewing_vector)
        great_circle_lat1, great_circle_lon1 = cartesian_to_lat_lon(*bone_vector1)

        great_circle_lat2, great_circle_lon2 = cartesian_to_lat_lon(*bone_vector2)

        global_viewing_angle_lat, global_viewing_angle_lon = cartesian_to_lat_lon(*viewing_vector)

        if last_projection_vector is None:
            last_projection_vector = viewing_vector
        last_proj_lat, last_proj_lon = cartesian_to_lat_lon(*last_projection_vector)

        view_cost_minizer.set_last_view_vector(last_projection_vector)

        answer_xyz = view_cost_minizer.minimize()

        num_points = 80
        latitudes = np.linspace(-90, 90, num_points)
        longitudes = np.linspace(-180, 180, num_points)

        # Generate the Cartesian coordinates and colors for each point on the sphere
        coords = []
        colors = []
        distances = []
        for lat in latitudes:
            for lon in longitudes:
                coords.append(lat_lon_to_cartesian(lat, lon))
                distances.append(lat_lon_to_distance(np.array([lat, lon])))

        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        for distance in distances:
            colors.append(cm.jet(distance))

        frame = {}
        frame['coords'] = coords
        frame['colors'] = colors
        frame['answer_xyz'] = answer_xyz
        frame['viewing_vector_xyz'] = lat_lon_to_cartesian(*cartesian_to_lat_lon(*viewing_vector))
        frame['last_projection_vector_xyz'] = lat_lon_to_cartesian(*cartesian_to_lat_lon(*last_projection_vector))

        frames.append(frame)

        last_projection_vector = answer_xyz

    def init():
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('Unit Sphere')

    def update(frame):
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        ax.clear()

        ax.scatter(*zip(*frame['coords']), s=5, c=frame['colors'], alpha=0.7)
        ax.scatter(*(1.1*np.array(frame['answer_xyz'])).tolist(), color='red', s=20, marker='x')
        ax.text(*(1.1*np.array(frame['answer_xyz'])).tolist(), "Minimum", color='black', zorder=2)

        # add the global viewing vector
        # viewing_vector_xyz = lat_lon_to_cartesian(*cartesian_to_lat_lon(*viewing_vector))
        ax.scatter(*(1.1*np.array(frame['viewing_vector_xyz'])).tolist(), color='green', s=20, marker='x')

        # add the last frame view vector
        # last_projection_vector_xyz = lat_lon_to_cartesian(*cartesian_to_lat_lon(*last_projection_vector))
        ax.scatter(*(1.1*np.array(frame['last_projection_vector_xyz'])).tolist(), color='yellow', s=20, marker='x')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ani = FuncAnimation(fig, update, frames=frames, init_func=init, repeat=True)
    plt.show()