import json
from typing import List

from tqdm import tqdm

from axifresco import Shape, Point


def bezier_to_catmull(vertices: List[Point]) -> Shape:
    catmull_vtx = [
        vertices[3]  + 6 * (vertices[0] - vertices[1]),
        vertices[0],
        vertices[3],
        vertices[0] + 6 (vertices[3] - vertices[2])
    ]

    return Shape(catmull_vtx, is_polygonal=False, ignore_ends=True)

def svg2shape(svg: str) -> List[Shape]:
    svg = svg.split(' ')
    svg = [instruction.split(',') for instruction in svg]

    class states:
        LINE = 'line'
        MOVE = 'move'
        BEZIER = 'bezier'
        HOME = 'home'
        UNKNOWN = 'unk'

    mapping = {
        'M': states.MOVE,
        'L': states.LINE,
        'C': states.BEZIER,
        'Z': states.HOME
    }

    shapes = []
    vertex_buffer = []
    state = states.LINE
    for instruction in svg:
        # switch state on instruction being preceded by a letter
        if len(instruction[0]) > 1:
            new_state = mapping.get(instruction[0][0], states.UNKNOWN)
            instruction[0] = instruction[0][1]

            # if switching to a bezier instruction, end the shape and create a new one
            # which will hold the bezier curve
            if new_state == states.BEZIER and state != states.BEZIER:
                if len(vertex_buffer) > 0:
                    shapes.append(Shape(vertex_buffer, True))
                vertex_buffer = [vertex_buffer[-1]]

        if state == states.MOVE:
            if len(vertex_buffer) > 0:
                shapes.append(Shape(vertex_buffer, True))
            vertex_buffer = [Point(float(instruction[0]), float(instruction[1]))]
            state = states.LINE
        elif state == states.LINE:
            vertex_buffer.append(Point(float(instruction[0]), float(instruction[1])))
        elif state == states.BEZIER:
            vertex_buffer.append((Point(float(instruction[0]), float(instruction[1]))))
            if len(vertex_buffer) == 4:
                shapes.append(bezier_to_catmull(vertex_buffer))
                vertex_buffer = [vertex_buffer[-1]]
        elif state == states.HOME:
            vertex_buffer.append(shapes[0].vertices[0])
        else:
            raise Exception(f'Encountered unkown state in svg description')

    if len(vertex_buffer) > 0:
        shapes.append(Shape(vertex_buffer, True)) 

    return shapes


def convert_font_to_catmull(json_file, font_name):
    with open(json_file, 'r') as f:
        svg_fonts = json.load(f)

    svg_font = svg_fonts[font_name]["chars"]

    class JSONPoint(dict):
        def __init__(self, point: Point):
            dict.__init__(self, x=point.x, y = point.y)    

    class JSONShape(dict):
        def __init__(self, shape: Shape):
            max_x = max([v.x for v in shape.vertices])
            min_x = min([v.x for v in shape.vertices])
            mid_x = 0.5 * (max_x + min_x)
            size_x = max_x - min_x
            max_y = max([v.y for v in shape.vertices])
            min_y = min([v.y for v in shape.vertices])
            mid_y = 0.5 * (max_y + min_y)
            size_y = max_y - min_y
            size = max(size_x, size_y)

            # center and normalize character, and make it serializable
            vertices = [JSONPoint((v - Point(mid_x, mid_y)) / size) for v in shape.vertices] 
            dict.__init__(
                self, vertices=vertices, is_polygonal=shape.is_polygonal,
                ignore_ends=shape.ignore_ends, canvas_width=1,
                canvas_height=1
            )

    for i, char in tqdm(enumerate(svg_font)):
        # convert char to Shape
        character = svg2shape(char['d'])
        # make into a json friendly format
        character = {'shapes' : [JSONShape(s) for s in character]}
        # export
        with open('./fonts/' + font_name + '_' + str(i) + '.json', 'w') as f:
            f.write(json.dumps(character))


if __name__ == '__main__':
    convert_font_to_catmull('./fonts/hersheytext.json', 'futural')
