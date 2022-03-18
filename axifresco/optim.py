from typing import Iterable, List, Tuple

import tqdm

from axifresco.axifresco import Shape, Point


def distSq_to_shape_ends(shape: Shape, point: Point) -> float:
    """Returns the square of the distance to a shape's
    ends from a given point
    """
    return min(point.distSq(shape.vertices[0]), point.distSq(shape.vertices[1]))


def reverse_segment_if_better(tour: List[Shape], i: int, j: int, k: int) -> List[Shape]:
    """If reversing tour[i:j] would make the tour shorter, then do it."""
    # Given tour [...A-B...C-D...E-F...]
    A, B, C, D, E, F = tour[i-1], tour[i], tour[j-1], tour[j], tour[k-1], tour[k % len(tour)]
    d0 = distSq_to_shape_ends(A, B) + distSq_to_shape_ends(C, D) + distSq_to_shape_ends(E, F)
    d1 = distSq_to_shape_ends(A, C) + distSq_to_shape_ends(B, D) + distSq_to_shape_ends(E, F)
    d2 = distSq_to_shape_ends(A, B) + distSq_to_shape_ends(C, E) + distSq_to_shape_ends(D, F)
    d3 = distSq_to_shape_ends(A, D) + distSq_to_shape_ends(E, B) + distSq_to_shape_ends(C, F)
    d4 = distSq_to_shape_ends(F, B) + distSq_to_shape_ends(C, D) + distSq_to_shape_ends(E, A)

    if d0 > d1:
        tour[i:j].reverse()
        return -d0 + d1
    elif d0 > d2:
        tour[j:k].reverse()
        return -d0 + d2
    elif d0 > d4:
        tour[i:k].reverse()
        return -d0 + d4
    elif d0 > d3:
        tmp = tour[j:k] + tour[i:j]
        tour[i:k] = tmp
        return -d0 + d3
    return 0


def closest_neighbour_sort(shapes: List[Shape]) -> List[Shape]:
    buffer = [shapes.pop(0)]

    for _ in tqdm(range(len(shapes) - 1)):
        # find closest shape to last shape in buffer
        closest_shape = min(
            shapes,
            key=lambda s: distSq_to_shape_ends(s, point=buffer[-1].vertices[-1])
        )

        # remove the shape from the list
        shapes.pop(shapes.index(closest_shape))

        # if closest point is the end of the shape, revert the shape
        if buffer[-1].vertices[-1].distSq(closest_shape.vertices[0]) > \
            buffer[-1].vertices[-1].distSq(closest_shape.vertices[-1]):
            closest_shape.vertices = [s for s in reversed(closest_shape.vertices)]
        buffer.append(closest_shape)

    # add the remaining shape in the list
    if buffer[-1].vertices[-1].distSq(shapes[0].vertices[0]) > \
        buffer[-1].vertices[-1].distSq(shapes[0].vertices[-1]):
        shapes[0].vertices = [s for s in reversed(shapes[0].vertices)]
    buffer.append(shapes[0])
    return buffer


def all_permutations(n: float) -> Iterable[Tuple[float]]:
    return ((i, j, k) for i in range(n) for j in range(i + 2, n) for k in range(j + 2, n + (i > 0)))


def optimize_path(shapes: List[Shape]) -> List[Shape]:
    """Sorts the shapes such that the amount of
    air time is minimized
    """
    # get a rough estimate by using closest nearest neighbour sort
    sorted_shapes = closest_neighbour_sort(shapes)

    delta = 1 
    while delta > 0:
        delta = 0
        for verts in all_permutations(len(shapes)):
            delta += reverse_segment_if_better(sorted_shapes, *verts)
            if delta > 0:
                break
    
    return sorted_shapes
