## МФТИ, студентка Ротцы АМ
## Код программы (`main.py`)
## python

import heapq
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

class Route:
    """Represents a route with aggregated metrics."""
    def __init__(self, path: List[int], length: int, time: int, cost: int):
        self.path = path
        self.length = length
        self.time = time
        self.cost = cost

def dijkstra(graph: Dict[int, Dict[int, Tuple[int, int, int]]], 
             start: int, 
             end: int, 
             weight_index: int) -> Optional[Route]:
    """
    Finds the shortest path using Dijkstra's algorithm by a single weight criterion.
    weight_index: 0 = length, 1 = time, 2 = cost.
    Returns a Route object or None if no path exists.
    """
    if start not in graph or end not in graph:
        return None

    distances = {node: float('inf') for node in graph}
    previous = {node: None for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        if current_dist != distances[current_node]:
            continue

        if current_node == end:
            break

        for neighbor, weights in graph[current_node].items():
            weight = weights[weight_index]
            new_dist = current_dist + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))

    if distances[end] == float('inf'):
        return None

    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()

    # Compute total metrics
    total_length = total_time = total_cost = 0
    for i in range(len(path) - 1):
        l, t, c = graph[path[i]][path[i+1]]
        total_length += l
        total_time += t
        total_cost += c

    return Route(path, total_length, total_time, total_cost)

def find_compromise_route(routes: List[Route], priorities: List[str]) -> Route:
    """Selects the best route based on priority criteria."""
    def is_better(r1: Route, r2: Route) -> bool:
        criteria = {
            'Д': (r1.length, r2.length),
            'В': (r1.time, r2.time),
            'С': (r1.cost, r2.cost)
        }
        for crit in priorities:
            v1, v2 = criteria[crit]
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
        return False  # equal

    best = routes[0]
    for r in routes[1:]:
        if is_better(r, best):
            best = r
    return best

def parse_input(filename: str = "input.txt"):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    sections = {'CITIES': [], 'ROADS': [], 'REQUESTS': []}
    current = None

    for line in lines:
        if line.startswith('[') and line.endswith(']'):
            current = line[1:-1]
        elif current in sections:
            sections[current].append(line)

    # Parse cities
    cities = {}
    for line in sections['CITIES']:
        cid, name = line.split(':', 1)
        cities[int(cid.strip())] = name.strip()

    city_to_id = {name: cid for cid, name in cities.items()}

    # Parse roads
    roads = []
    for line in sections['ROADS']:
        road_part, params_part = line.split(':', 1)
        ids = road_part.replace('-', ' ').split()
        id1, id2 = int(ids[0]), int(ids[1])
        params = list(map(int, params_part.split(',')))
        roads.append((id1, id2, params[0], params[1], params[2]))

    # Parse requests
    requests = []
    for line in sections['REQUESTS']:
        route_part, prio_part = line.split('|', 1)
        from_city, to_city = [s.strip() for s in route_part.split('->')]
        priorities = [p.strip() for p in prio_part.strip('()').split(',')]
        requests.append((from_city, to_city, priorities))

    return cities, city_to_id, roads, requests

def build_graph(roads: List[Tuple[int, int, int, int, int]]) -> Dict[int, Dict[int, Tuple[int, int, int]]]:
    graph = defaultdict(dict)
    for id1, id2, length, time, cost in roads:
        graph[id1][id2] = (length, time, cost)
        graph[id2][id1] = (length, time, cost)
    return graph

def format_route(route: Route, cities: Dict[int, str]) -> str:
    return " -> ".join(cities[node] for node in route.path)

def main():
    try:
        cities, city_to_id, roads, requests = parse_input()
        graph = build_graph(roads)

        results = []

        for from_name, to_name, priorities in requests:
            from_id = city_to_id.get(from_name)
            to_id = city_to_id.get(to_name)

            if from_id is None or to_id is None:
                print(f"Ошибка: город '{from_name}' или '{to_name}' не найден.", file=sys.stderr)
                continue

            # Find optimal routes by each criterion
            r_length = dijkstra(graph, from_id, to_id, 0)
            r_time   = dijkstra(graph, from_id, to_id, 1)
            r_cost   = dijkstra(graph, from_id, to_id, 2)

            if not any([r_length, r_time, r_cost]):
                print(f"Нет маршрута между {from_name} и {to_name}", file=sys.stderr)
                continue

            routes = {'ДЛИНА': r_length, 'ВРЕМЯ': r_time, 'СТОИМОСТЬ': r_cost}
            results.append((routes, priorities, cities))

        # Write output
        with open("output.txt", "w", encoding="utf-8") as f:
            for i, (routes, priorities, cities) in enumerate(results):
                for key in ['ДЛИНА', 'ВРЕМЯ', 'СТОИМОСТЬ']:
                    r = routes[key]
                    if r:
                        path_str = format_route(r, cities)
                        f.write(f"{key}: {path_str} | Д={r.length}, В={r.time}, С={r.cost}\n")
                    else:
                        f.write(f"{key}: Не найдено\n")

                valid_routes = [r for r in routes.values() if r]
                compromise = find_compromise_route(valid_routes, priorities)
                path_str = format_route(compromise, cities)
                f.write(f"КОМПРОМИСС: {path_str} | Д={compromise.length}, В={compromise.time}, С={compromise.cost}\n")

                if i < len(results) - 1:
                    f.write("\n")

        print("Результат записан в output.txt")

    except FileNotFoundError:
        print("Файл input.txt не найден.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
