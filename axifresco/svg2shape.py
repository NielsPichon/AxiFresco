import sys
import json
from typing import List
from dataclasses import asdict
from xml.dom import minidom

import matplotlib.pyplot as plt

from svgpathtools import svg2paths, Line, CubicBezier, Arc, QuadraticBezier 

from axifresco.axifresco import Shape, Point


def bezier_to_catmull(vertices: List[Point]) -> Shape:
    catmull_vtx = [
        vertices[3]  + 6 * (vertices[0] - vertices[1]),
        vertices[0],
        vertices[3],
        vertices[0] + 6 * (vertices[3] - vertices[2])
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
            if instruction[0][0].isalpha():
                new_state = mapping.get(instruction[0][0], states.UNKNOWN)
                instruction[0] = instruction[0][1:]

                # if switching to a bezier instruction, end the shape and create a new one
                # which will hold the bezier curve
                if new_state == states.BEZIER and state != states.BEZIER:
                    if len(vertex_buffer) > 0:
                        shapes.append(Shape(vertex_buffer, True))
                    vertex_buffer = [vertex_buffer[-1]]
                state = new_state

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

def paths2shapes(path):
    shapes = []
    vertex_buffer = []
    for move in path:
        if isinstance(move, Line):
            # if there already is a vertex in the buffer,
            # check that the shape is continuous.
            if len(vertex_buffer) > 0:
                # if there is a discontinuity, store the vertex buffer
                # as a shape and start with a fresh buffer
                if move.start != vertex_buffer[-1]:
                    shapes.append(Shape([Point.from_complex(p) for p in vertex_buffer], is_polygonal=True))
                    vertex_buffer = [move.start]
            vertex_buffer.append(move.end)
        elif isinstance(move, CubicBezier):
            # if the buffer is not empty, register it as a shape
            if len(vertex_buffer) > 0:
                shapes.append(Shape([Point.from_complex(p) for p in vertex_buffer], is_polygonal=True))
                vertex_buffer = []
            # convert the bezier to a catmull-Rom shape
            shapes.append(
                bezier_to_catmull([
                    Point.from_complex(move.start),
                    Point.from_complex(move.control1),
                    Point.from_complex(move.control2),
                    Point.from_complex(move.end)
                ])
            )
        elif isinstance(move, QuadraticBezier):
            raise NotImplementedError('Quadratic Bezier are not yet supported')
        elif isinstance(move, Arc):
            raise NotImplementedError('Arcs are not yet supported')
    
    if len(vertex_buffer) > 0:
        shapes.append(Shape([Point.from_complex(p) for p in vertex_buffer], is_polygonal=True))

    return shapes

def convert_svg_file(filepath: str):
    # find drawing width and height
    svg_xml = minidom.parse(filepath)
    root = svg_xml.documentElement
    width = int(root.getAttribute('width'))
    height = int(root.getAttribute('height'))
    canvasSize = Point(width, height)

    # convert svg to paths
    paths, _ = svg2paths(filepath)

    # convert paths to shapes
    shapes = []
    for path in paths:
        shapes.extend(paths2shapes(path))

    # normalize coordinates
    for s in shapes:
        for i, v in enumerate(s.vertices):
            s.vertices[i] = v / canvasSize

    # convert to json
    shapes_svg = {'shapes': [asdict(s) for s in shapes]}
    for s in shapes_svg['shapes']:
        s['canvas_width'] = width
        s['canvas_height'] = height

    return json.dumps(shapes_svg)    

def convert_font_to_catmull(json_file, font_name):
    with open(json_file, 'r') as f:
        svg_fonts = json.load(f)

    svg_font = svg_fonts[font_name]["chars"]

    class JSONPoint(dict, Point):
        def __init__(self, point: Point):
            dict.__init__(self, x=point.x, y = point.y)
            Point.__init__(self, point.x, point.y)

    class JSONShape(dict, Shape):
        def __init__(self, shape: Shape):
            # center and normalize character, and make it serializable
            vertices = [JSONPoint(v) for v in shape.vertices] 
            dict.__init__(
                self, vertices=vertices, is_polygonal=shape.is_polygonal,
                ignore_ends=shape.ignore_ends, canvas_width=1,
                canvas_height=1
            )
            Shape.__init__(self, vertices, shape.is_polygonal, shape.ignore_ends)

    characters = {}
    for i, char in enumerate(svg_font[:-1]):
        letter = (i + 33).to_bytes(1, 'little').decode('utf-8')
        # convert char to Shape
        character = svg2shape(char['d'])
        if len(sys.argv) > 1 and bool(sys.argv[1]) == True:
            img = None
            for s in character:
                img = s.preview(img, 10, center=False)
            plt.imshow(img)
            plt.show()
            img.close()
        # make into a json friendly format
        character = {'shapes' : [JSONShape(s) for s in character], 'width': int(char['o'])}
        characters[letter] = character
    print('Supported character:')
    print(''.join(list(characters.keys())))
    # export
    with open('./fonts/' + font_name + '.json', 'w') as f:
        f.write(json.dumps(characters))


if __name__ == '__main__':
    # convert_font_to_catmull('./fonts/hersheytext.json', 'futural')
    data = convert_svg_file('C://Users/niels/Desktop/recto.svg')
    with open('recto.json', 'w') as f:
        f.write(data)
