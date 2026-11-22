import numpy as np
import matplotlib.pyplot as plt
import stim
import pymatching
import sinter
from typing import List
import os
import math
import networkx
import heapq


class graph_neighbor():
    def __init__(self, nname, prob, obs_mask=0):
        self.nname = nname          # index of the neighbouring node (num_dets = boundary)
        self.prob = prob            # probability that this edge flips
        self.obs_mask = obs_mask    # bitmask of logical observables this edge flips
        self.weight = math.log((1 - prob) / prob)


class sim_graph():
    """Matching graph built from a (decomposed) detector error model.

    Nodes 0..size-1 are detectors, node `size` is the virtual boundary.
    """

    def __init__(self, size, dem):
        self.size = size
        self.boundary = size
        edges = self.parse_input(size, dem)
        self.adj = self.construct_graph(edges, size)
        self.location = dem.get_detector_coordinates()

    def parse_input(self, size, dem):
        edges = {}
        for inst in dem.flattened():
            if inst.type != "error":
                continue  # detector / logical_observable lines carry no edges
            prob = inst.args_copy()[0]
            if prob <= 0:
                continue
            for group in inst.target_groups():
                dets = [t.val for t in group if t.is_relative_detector_id()]
                obs_mask = 0
                for t in group:
                    if t.is_logical_observable_id():
                        obs_mask ^= 1 << t.val
                if len(dets) == 1:
                    key = (dets[0], self.boundary)
                elif len(dets) == 2:
                    key = (min(dets), max(dets))
                else:
                    continue  # not graphlike (shouldn't happen with decompose_errors=True)
                edges.setdefault(key, []).append((prob, obs_mask))
        return edges

    def construct_graph(self, edges, size):
        adj = [[] for _ in range(size + 1)]
        for (u, v), contributions in edges.items():
            # Combine independent mechanisms on the same edge: the edge flips
            # if an odd number of them fire.
            p = 0.0
            for p_i, _ in contributions:
                p = p * (1 - p_i) + p_i * (1 - p)
            if p <= 0 or p >= 0.5:
                continue
            # Parallel mechanisms should agree on observables; take the most likely one's.
            obs_mask = max(contributions)[1]
            adj[u].append(graph_neighbor(v, p, obs_mask))
            adj[v].append(graph_neighbor(u, p, obs_mask))
        return adj


def dijkstra(this_error, errors, adj, boundary):
    """Shortest paths from `this_error` to every node in `errors` and to the boundary.

    Returns {node: (distance, obs_mask_along_path)} for the targets that are reachable.
    The boundary is treated as a sink: paths never pass *through* it.
    """
    targets = set(errors)
    targets.add(boundary)
    targets.discard(this_error)

    dist = {this_error: 0.0}
    done = set()
    found = {}
    pq = [(0.0, this_error, 0)]
    while pq and len(found) < len(targets):
        d, node, obs = heapq.heappop(pq)
        if node in done:
            continue
        done.add(node)
        if node in targets:
            found[node] = (d, obs)
        if node == boundary:
            continue
        for nb in adj[node]:
            nd = d + nb.weight
            if nb.nname not in done and nd < dist.get(nb.nname, math.inf):
                dist[nb.nname] = nd
                heapq.heappush(pq, (nd, nb.nname, obs ^ nb.obs_mask))
    return found


def mwpm(errors, apsp, boundary_dist):
    """Minimum-weight perfect matching on the standard boundary-augmented graph.

    Each defect i gets a boundary copy ('b', i). Edges:
      i -- j        weight apsp[i][j]
      i -- ('b', i) weight boundary_dist[i]
      ('b', i) -- ('b', j) weight 0
    Returns a list of pairs (i, j) of defect indices, with j = None meaning "matched to boundary".
    """
    k = len(errors)
    G = networkx.Graph()
    edges = []
    for i in range(k):
        if boundary_dist[i] < math.inf:
            edges.append((i, ('b', i), boundary_dist[i]))
        for j in range(i + 1, k):
            if apsp[i][j] < math.inf:
                edges.append((i, j, apsp[i][j]))
            edges.append((('b', i), ('b', j), 0.0))
    # Every perfect matching has exactly k edges, so maximising (C - w) among
    # max-cardinality matchings is the same as minimising w.
    C = 1.0 + max((w for _, _, w in edges), default=0.0)
    for u, v, w in edges:
        G.add_edge(u, v, weight=C - w)
    matching = networkx.max_weight_matching(G, maxcardinality=True)

    pairs = []
    for u, v in matching:
        u_real, v_real = isinstance(u, int), isinstance(v, int)
        if u_real and v_real:
            pairs.append((u, v))
        elif u_real:
            pairs.append((u, None))
        elif v_real:
            pairs.append((v, None))
        # boundary-copy <-> boundary-copy edges mean nothing physically
    return pairs


class MyCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, dem: stim.DetectorErrorModel):
        self.num_dets = dem.num_detectors
        self.num_obs = dem.num_observables
        self.dem = dem
        self.graph = sim_graph(self.num_dets, dem)

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,   # uint8, shape (shots, ceil(num_dets / 8))
    ) -> np.ndarray:                                    # uint8, shape (shots, ceil(num_obs / 8))
        # unpack to bool: shape (shots, num_dets)
        dets = np.unpackbits(bit_packed_detection_event_data,
                             axis=1, bitorder="little")[:, :self.num_dets].astype(bool)

        pred = np.zeros((dets.shape[0], self.num_obs), dtype=bool)
        for i, shot in enumerate(dets):
            pred[i] = self.decode_one(shot)

        return np.packbits(pred, axis=1, bitorder="little")

    def decode_one(self, shot: np.ndarray) -> np.ndarray:
        # shot: bool array of length num_dets (True = lit detector)
        # return: bool array of length num_obs (True = flip that logical)
        errors = np.flatnonzero(shot).tolist()
        k = len(errors)
        if k == 0:
            return np.zeros(self.num_obs, dtype=bool)

        adj = self.graph.adj
        boundary = self.graph.boundary

        apsp = np.full((k, k), math.inf)
        apsp_obs = np.zeros((k, k), dtype=np.int64)
        boundary_dist = np.full(k, math.inf)
        boundary_obs = np.zeros(k, dtype=np.int64)
        index = {e: idx for idx, e in enumerate(errors)}

        for i, e in enumerate(errors):
            found = dijkstra(e, errors, adj, boundary)
            for node, (d, obs) in found.items():
                if node == boundary:
                    boundary_dist[i] = d
                    boundary_obs[i] = obs
                else:
                    j = index[node]
                    apsp[i][j] = d
                    apsp_obs[i][j] = obs

        obs_flip = 0
        for i, j in mwpm(errors, apsp, boundary_dist):
            obs_flip ^= int(boundary_obs[i]) if j is None else int(apsp_obs[i][j])
        print(self.num_obs)
        return np.array([(obs_flip >> b) & 1 for b in range(self.num_obs)], dtype=bool)


class MyDecoder(sinter.Decoder):
    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return MyCompiledDecoder(dem)


def main():
    a = run_sinter_benchmark(drange=[3, 5])
    analyze_surface_code_stats(a)


# Benchmarking
def run_sinter_benchmark(mydecfn=None, noise=1e-3, drange=[3, 5, 7, 9],
                         max_shots=2_000, max_errors=100):
    if mydecfn is None:
        mydecfn = MyDecoder()
    surface_code_tasks = [
        sinter.Task(
            circuit=stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=d * 3,
                distance=d,
                after_clifford_depolarization=noise,
                after_reset_flip_probability=noise,
                before_measure_flip_probability=noise,
                before_round_data_depolarization=noise,
            ),
            json_metadata={'d': d, 'r': d * 3, 'p': noise},
        )
        for d in drange
    ]

    collected_surface_code_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=surface_code_tasks,
        decoders=["mine", "pymatching"],          # names to run
        custom_decoders={"mine": mydecfn},
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=False,
    )
    return collected_surface_code_stats


def analyze_surface_code_stats(collected_surface_code_stats):
    for stats in sorted(collected_surface_code_stats,
                        key=lambda s: (s.json_metadata['d'], s.decoder)):
        d = stats.json_metadata['d']
        if not stats.errors:
            print(f"[{stats.decoder}] Didn't see any errors for d={d} ({stats.shots} shots)")
            continue
        per_shot = stats.errors / stats.shots
        per_round = sinter.shot_error_rate_to_piece_error_rate(per_shot, pieces=stats.json_metadata['r'])
        print(f"[{stats.decoder}] d:{d}, shots {stats.shots}, errors {stats.errors}, per round {per_round:.3e}")


if __name__ == "__main__":
    main()

"""
My impelmentation, I think I understand this part
claude implemented APSP and the matching algo cuz its not really qc

It also made my code cleaner


import numpy as np
import matplotlib.pyplot as plt
import stim
import pymatching
import sinter
from typing import List
imp
impor
import scipy
import re
import networkx
import heapq


pattern = re.compile(
    r"detector\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)\s*D(\d+)"
)

class graph_neighbor():
    def __init__(self, nname, prob):
        self.nname = nname
        self.prob = prob

class sim_graph():
    def __init__(self, size, dem):
        adj_mat, location = self.parse_input(size,dem)
        self.adj_mat = self.construct_graph(adj_mat, size)



    def parse_input(self, size, dem):
        adj_list = ()
        adj_mat = [[]] * size * size
        location = [None] * size

        for line in dem.flattened():
            if line[0] == "e":
                data  = line.split(" ")
                prob = float(data[0][6:-1])
                all_nodes = []
                cur_pair = []
                for i in range(1,len(data)):
                    if(data[i] == "^"):
                        all_nodes.append(cur_pair)
                    else:
                        cur_pair.append(data[i])
                all_nodes.append(cur_pair)
                
                for pair in all_nodes:
                    if(len(pair) != 2): 
                        continue
                    idx1 = int(pair[0][1:])
                    idx2 = int(pair[1][1:])
                    adj_mat[size * idx1 + idx2].append(prob)
                    adj_mat[size * idx2 + idx1].append(prob)
            elif line[0] == "d":
                m = pattern.search(line)
                a, b, c = map(float, m.groups()[:3])   # 2.0, 4.0, 8.0
                d = int(m.group(4))                    # 65
                location[d] = (a,b,c)
            else:
                print("parse error")
        return adj_mat, location

    def construct_graph(self, adj_mat,size):
        adj_new = np.zeros(adj_mat, size)
        for x in range(size):
            for y in range(size):
                p1 = adj_new[x,y]
                p2 = adj_mat[x*size, y]
                adj_new[x,y] = p1(1-p2) + p2(1-p1)
        return adj_new

# claude's formula:
# p = 0.0
# for p_i in contributions:
#     p = p * (1 - p_i) + p_i * (1 - p)

class MyCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, dem: stim.DetectorErrorModel):
        self.num_dets = dem.num_detectors
        self.num_obs = dem.num_observables
        self.dem = dem
        dists = [3,5,7,9,11]
        for d in dists:
            if(self.num_dets == 3 * d *(d * d - 1)):
                self.dist = d
                break
        self.graph = sim_graph(self.num_dets, dem)



    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,   # uint8, shape (shots, ceil(num_dets / 8))
    ) -> np.ndarray:                                    # uint8, shape (shots, ceil(num_obs / 8))
        # unpack to bool: shape (shots, num_dets)
        dets = np.unpackbits(bit_packed_detection_event_data,
                             axis=1, bitorder="little")[:, :self.num_dets].astype(bool)

        pred = np.zeros((dets.shape[0], self.num_obs), dtype=bool)
        for i, shot in enumerate(dets):
            pred[i] = self.decode_one(shot)            # your decoder: detectors -> observable flips

        return np.packbits(pred, axis=1, bitorder="little")

    def decode_one(self, shot: np.ndarray) -> np.ndarray:
        # shot: bool array of length num_dets (True = lit detector)
        # return: bool array of length num_obs (True = flip that logical)
        measurements_per = self.dist * self.dist - 1

        self.graph.adj_mat

        errors = [i for i, val in enumerate(shot) if val]
        apsp = -1 * np.ones_like((len(errors), len(errors)))
        for error in errors:


        

        coords = self.dem.get_detector_coordinates()  
        print(f"{len(coords)}, {self.num_dets}")
        return np.zeros(self.num_obs, dtype=bool)

def dijkstra(this_error, errors, adj):
    d = dict()
    for idx, e in enumerate(errors):
        d[e] = idx
    path = [-1 ] * len(errors)
    pq = []
    heapq.push(pq,(0, this_error))
    while(len(pq) > 0):
        priority, task = heapq.heappop(pq)
        if(path[d[task]] != -1):
            continue
            



class MyDecoder(sinter.Decoder):
    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return MyCompiledDecoder(dem)


def main():
    a = run_sinter_benchmark(drange = [3])
    analyze_surface_code_stats(a)



    pass




# Benchmarking
def run_sinter_benchmark(mydecfn = MyDecoder(), noise=1e-3, drange = [3, 5, 7, 9]):
    surface_code_tasks = [
        sinter.Task(
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=d * 3,
                distance=d,
                after_clifford_depolarization=noise,
                after_reset_flip_probability=noise,
                before_measure_flip_probability=noise,
                before_round_data_depolarization=noise,
            ),
            json_metadata={'d': d, 'r': d * 3, 'p': noise},
        )
        for d in drange
    ]

    collected_surface_code_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=surface_code_tasks,
        decoders=["mine", "pymatching"],          # names to run
        custom_decoders={"mine": mydecfn},        
        max_shots=1,
        max_errors=1,
        print_progress=False,
    )
    return collected_surface_code_stats

def analyze_surface_code_stats(collected_surface_code_stats):
    for stats in collected_surface_code_stats:
        d = stats.json_metadata['d']
        if not stats.errors:
            print(f"Didn't see any errors for d={d}")
            continue
        per_shot = stats.errors / stats.shots
        per_round = sinter.shot_error_rate_to_piece_error_rate(per_shot, pieces=stats.json_metadata['r'])
        print(f"d:{d}, per round {per_round}")

if __name__ == "__main__":
    main()  per_round = sinter.shot_error_rate_to_piece_error_rate(per_shot, pieces=stats.json_metadata['r'])
        print(f"d:{d}, per round {per_round}")

if __name__ == "__main__":
    main()
    """
