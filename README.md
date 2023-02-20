from copy import deepcopy
import math
from typing import List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from guide.models import Point

MULTIPLIER = 0.8
EARTH_RADIUS = 6372.800   # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
KMH_TO_MS_MULTIPLIER = 3.6
# WALK_SPEED = 2.5 * KMH_TO_MS_MULTIPLIER


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * EARTH_RADIUS


class GraphPoint(object):
    def __init__(self, point: 'Point', neighbours: List['GraphPoint'] = None):
        self.point = point
        self.x_coord = self.point.x_coord
        self.y_coord = self.point.y_coord
        self.neighbours = {x: self.get_distance(x) for x in neighbours} if neighbours else dict()

    def get_distance(self, other: 'GraphPoint') -> float:
        """Get distance in meters."""
        return haversine(self.x_coord, self.y_coord, other.x_coord, other.y_coord) * 1000

    def add_neighbour(self, neighbour: 'GraphPoint') -> None:
        if neighbour == self:
            return
        if neighbour not in self.neighbours:
            self.neighbours[neighbour] = self.get_distance(neighbour)

    def __str__(self) -> str:
        return f"GraphPoint: {self.x_coord}, {self.y_coord}"

    def is_equal(self, other: 'GraphPoint') -> bool:
        return self.x_coord == other.x_coord and self.y_coord == other.y_coord

    def is_this_point(self, p: 'Point') -> bool:
        return self.point is p


class GraphRouteNode(object):
    def __init__(self, point: GraphPoint, distance: float):
        self.point = point
        self.distance = distance

    def __str__(self) -> str:
        return f"{self.point} ({self.distance})"


class GraphRoute(object):
    def __init__(self, points: List[GraphRouteNode] = None):
        self.nodes = points if points else list()

    def add_node(self, node: GraphRouteNode) -> None:
        self.nodes.append(node)

    def add_point(self, point: GraphPoint, distance: float = None) -> None:
        if not distance:
            distance = 0 if not len(self.nodes) else point.get_distance(self.nodes[-1].point)
        self.add_node(GraphRouteNode(point, distance))

    def __str__(self) -> str:
        return f"GraphRoute {self.length()}: " + " -> ".join([str(x) for x in self.nodes])

    def point_in_route(self, point) -> bool:
        for p in self.nodes:
            if p.point.is_equal(point):
                return True
        return False

    def length(self) -> float:
        return sum([x.distance for x in self.nodes])


class Graph(object):
    def __init__(self, points: List[GraphPoint], entry_point: GraphPoint):
        self.entry_point = entry_point
        self.points = points
        self.init_distances()

    def init_distances(self) -> None:
        for i in range(1, len(self.points)):
            for j in range(i):
                self.points[i].add_neighbour(self.points[j])
                self.points[j].add_neighbour(self.points[i])

    def routes(self, length: float, new_point: GraphPoint = None,
               parent_route: GraphRoute = None) -> GraphRoute | List[GraphRoute] | None:
        if not new_point:
            new_point = self.entry_point
        if not parent_route:
            parent_route = GraphRoute()
        new_route = deepcopy(parent_route)
        new_route.add_point(new_point)
        if length * MULTIPLIER < new_route.length() < length:
            return new_route
        if new_route.length() > length:
            return None
        to_check = [x for x in new_point.neighbours if not new_route.point_in_route(x)]
        _routes = [self.routes(length, x, new_route) for x in to_check]

        res = list()
        for _route in _routes:
            if _route is not None:
                if type(_route) is GraphRoute:
                    res.append(_route)
                elif type(_route) is list:
                    res += _route
                else:
                    raise TypeError(f'Wrong route[s] type: {type(_route)}')
        return res


def find_routes(points: List['Point'], entry_point: 'Point', route_length: Union[int, float],
                required_point: 'Point' = None) -> List[GraphRoute]:
    graph_points = [GraphPoint(x) for x in points]
    entry_point = [x for x in graph_points if x.is_this_point(entry_point)][0]
    if required_point:
        required_point = [x for x in graph_points if x.is_this_point(required_point)][0]

    graph = Graph(graph_points, entry_point)
    routes = graph.routes(route_length)

    if required_point:
        routes = [x for x in routes if x.point_in_route(required_point)]

    return routes
