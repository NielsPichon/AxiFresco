import argparse
import atexit
import os
import time
import json
from functools import partial
from typing import Dict, List, NoReturn, Tuple
from dataclasses import dataclass
from multiprocessing import Queue
from threading import Event


from tqdm import tqdm
from pyaxidraw import axidraw
import colorama
from colorama import Fore
from pynput import keyboard
from natsort import natsorted
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt 


colorama.init(autoreset=True)


OPTIONS=[
    'speed_pendown',
    'speed_penup',
    'accel',
    'pen_pos_down',
    'pen_pos_up',
    'pen_rate_lower',
    'pen_rate_raise',
    'pen_delay_down',
    'pen_delay_up',
    'const_speed',
    'model',
    'port',
    'port_config',
    'units' 
]

ignored = ['units']

@dataclass
class Point:
    """
    A utility class storing a point with basic 2D vector arithmetic. Units are in mm.
    """

    x: float = 0
    y: float = 0

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __add__(self, b):
        if isinstance(b, Point):
            return Point(self.x + b.x, self.y + b.y)
        else:
            return Point(self.x + b, self.y + b)
    
    def __radd__(self, b):
        return self + b

    def __sub__(self, b):
        if isinstance(b, Point):
            return Point(self.x - b.x, self.y - b.y)
        else:
            return Point(self.x - b, self.y - b)
    
    def __rsub__(self, b):
        return -self + b

    def __mul__(self, b):
        if isinstance(b, Point):
            return Point(self.x * b.x, self.y * b.y)
        else:
            return Point(self.x * b, self.y * b)
    
    def __rmul__(self, b):
        return self * b
    
    def __truediv__(self, b):
        if isinstance(b, Point):
            return Point(self.x / b.x, self.y / b.y)
        else:
            return Point(self.x / b, self.y / b) 

    def __iter__(self):
        return [self.x, self.y]

    def __eq__(self, o) -> bool:
        if isinstance(o, Point):
            return self.x == o.x and self.y == o.y
        else:
            raise Exception(f"Cannot compare {type(o)} and {Point}")

    def distSq(self, point) -> float:
        direction = point - self
        return direction.x ** 2 + direction.y ** 2


class FORMATS:
    A3 = Point(297, 420)
    A4 = Point(210, 297)
    A5 = Point(148, 210)


@dataclass
class Shape:
    vertices: List[Point]
    is_polygonal: bool
    ignore_ends: bool = False

    def catmull_rom(self, p0: Point, p1: Point, p2: Point, p3: Point) -> List[float]:
        """
        Computes the 3rd order polynomial coefficients describing a
        Catmull Rom spline with alpha=0.5 passing through
        the 4 specified points"""
        a = 0.5 * (-p0 + 3 * p1 - 3 * p2 + p3)
        b = 0.5 * (2 * p0 - 5 * p1 + 4 * p2 - p3)
        c = 0.5 * (-p0 + p2)
        d = 0.5 * (2 * p1)
        return a, b, c, d

    def get_segment(self, idx: int, resolution: int = 10) -> List[Point]:
        """
        Returns a list of points describing the edge
        starting at the specified point.
        * If the shape is polygonal, this will return a list
        containing the point and the next one.
        * If the shape is made of spline, this will return an
        array of points along the Catmull-Rom spline. 
        """
        # if starting at last point, of the spline or further,
        # don't return anything
        if idx >= len(self.vertices) - 1:
            return []

        # else if polygonal, simply return the point at the
        # specified index and the next one
        if self.is_polygonal:
            return [self.vertices[idx], self.vertices[idx + 1]]
        # else if made of splines, retrieve control points of the spline,
        # that is the point, the one before and the 2 after.
        # If close to the end of the shape, the first and last control points will
        # either be clamped to the shape end, or loop along the shape if the shape
        # is closed
        else:
            if idx == 0:
                if self.vertices[0] == self.vertices[-1]:
                    p0 = self.vertices[-2]
                else:
                    p0 = self.vertices[0]
            else:
                p0 = self.vertices[idx - 1]
            
            p1 = self.vertices[idx]
            p2 = self.vertices[idx + 1]

            if idx == len(self.vertices) - 2:
                if self.vertices[0] == self.vertices[-1]:
                    p3 = self.vertices[1]
                else:
                    p3 = self.vertices[-1]
            else:
                p3 = self.vertices[idx + 2]

            # compute spline coefficients
            a, b, c, d = self.catmull_rom(p0, p1, p2, p3)
            # create 100 gradutations along the spline
            pts = [self.vertices[idx]]
            for t in list(range(resolution))[1:-1]:
                t = t / resolution
                pts.append(d + t * (c + t * (b + t * a)))
            pts.append(self.vertices[idx + 1])

            return pts

    def preview(self, img: Image = None, scale: float = 1, center: bool = True) -> Image:
        """
        Draw the shape using PIL. If an img is provided as
        argument, use this one. Else create a new one.
        """
        if img is None:
            img = Image.new('RGB', (1000, 1000), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        if len(self.vertices) > 0:
            x = []
            y = []
            for idx in range(len(self.vertices)):
                if self.ignore_ends and idx == 0 or idx == len(self.vertices) - 1:
                    continue
                points = self.get_segment(idx, 10)

                for vtx in points:
                    x.append(vtx.x * scale)
                    y.append(vtx.y * scale)

            if center:
                x = [500 + (xx - (max(x) + min(x)) * 0.5) for xx in x]
                y = [500 + (yy - (max(y) + min(y)) * 0.5) for yy in y]
            
            xy = []
            for xx, yy in zip(x, y):
                xy.append(xx)
                xy.append(yy)

            draw.line(xy, fill=(255, 255, 255), width=1)

        return img


class Axifresco:
    """
    The main class to handle communication with the axidraw 
    """

    def __init__(self, config: Dict = None, reset=False, resolution:int = 10) -> None:
        self.resolution = resolution

        # initialise axidraw API
        self.axidraw = axidraw.AxiDraw()
        self.axidraw.interactive()

        # init pause toggle
        self.pause = Event()
        #init abort toggle
        self.abort = Event()

        # set configuration
        if config is None:
            config = {}

        # Force units to be mm
        config['units'] = 2

        self.set_config(config)

        # init axidraw state
        self.is_connected = False
        self.position = Point(0, 0)

        # set default fromat to A3
        self.format = FORMATS.A3

        if not reset:
            # make sure we disconnect before exiting
            atexit.register(self.close)

            print(Fore.GREEN + 'Please move the axidraw to the home position '
                'and press a key to continue...')
            input()

    def error(self, text: str) -> None:
        if not isinstance(text, str):
            text = str(text)
        print(Fore.RED + text)

    def set_format(self, format: Point) -> None:
        self.format = format

    def set_config(self, config: Dict) -> bool:
        """
        Allows setting the axidraw options
        """

        try:
            for key, value in config.items():
                if key in OPTIONS:
                    exec(f'self.axidraw.options.{key} = {value}')
            self.axidraw.update()
        except Exception as e:
            self.error(e)
            return False

        return True
    
    def do_action(func):
        """
        Wrapper for any action to perform on the axidraw. Before performaing the action,
        user approval will be requested if relevant and connection/disconnection to
        the axidraw will be performed accordingly.
        """
        def action(self, *args, **kwargs) -> bool:
            ask_verification = kwargs.pop('ask_verification', False)
            disconnect_on_end = kwargs.pop('disconnect_on_end', False)
            allow_pause = kwargs.pop('allow_pause', False)
            go_home = kwargs.pop('go_home', False)

            if ask_verification:
                print(Fore.GREEN + f'Press any key to proceed with {func.__name__}...')
                input()

            if self.connect():
                if allow_pause:
                    # create a thread which listens for key presses and pauses the
                    # draw process accordingly
                    def on_press(key, pause_event, abort_event):
                        if not abort_event.is_set():
                            if key == keyboard.Key.space:
                                if pause_event.is_set():
                                    print('Resuming draw.')
                                    pause_event.clear()
                                else:
                                    pause_event.set()
                                    print('Pause... Press [space] to resume or [escape] to abort.')
                            if key == keyboard.Key.esc and pause_event.is_set():
                                abort_event.set()

                    key_thread = keyboard.Listener(
                        on_press=partial(on_press, pause_event=self.pause, abort_event=self.abort))

                    print('Pause the drawing at any point by pressing [SPACEBAR]')
                    key_thread.start()

                ret = func(self, *args, **kwargs)

                if allow_pause:
                    # is_done.set()
                    key_thread.stop()
                    key_thread.join()

                if go_home:
                    ret = ret or self.move_home()

                if disconnect_on_end:
                    self.close()

                return ret
            else:
                return False

        return action

    def connect(self) -> bool:
        if not self.is_connected:
            try:
                ret = self.axidraw.connect()
                if ret:
                    self.is_connected = True
                    position = self.axidraw.current_pos()
                    self.position = Point(*position)
                return ret
            except Exception as e:
                self.error(e)
                return False
        else:
            return True

    @do_action
    def move_home(self) -> bool:
        try:
            return self.move_to(Point(0, 0))
        except Exception as e:
            self.error(e)
            return False

    @do_action
    def pen_up(self) -> bool:
        try:
            self.axidraw.penup()
        except Exception as e:
            self.error(e)
            return False
        return True
    
    @do_action
    def pen_down(self) -> bool:
        try:
            self.axidraw.pendown()
        except Exception as e:
            self.error(e)
            return False
        return True

    @do_action
    def move_to(self, point: Point) -> bool:
        if self.position != point:
            try:
                self.wait_for_resume()
                self.axidraw.moveto(point.y, point.x)
                self.position = point
            except Exception as e:
                self.error(e)
                return False
        return True

    @do_action
    def line_to(self, point: Point) -> bool:
        try:
            self.wait_for_resume()
            self.axidraw.lineto(point.y, point.x)
            self.position = point
            if self.pause.is_set():
                self.pen_up()
        except Exception as e:
            self.error(e)
            return False
        return True

    def wait_for_resume(self) -> None:
        while self.pause.is_set():
            if self.abort.is_set():
                print('Aborting...')
                try:
                    self.pause.clear()
                    self.move_home()
                    self.close()
                except:
                    print('Something went wrong when aborting')
                exit()
            time.sleep(0.1)

    @do_action
    def draw_shape(self, shape: Shape) -> bool:
        # quickly go through all points and make sure are within bounds of the canvas
        if not shape.is_polygonal:
            self.set_config({'const_speed': True})
        else:
            self.set_config({'const_speed': False})
        for point in shape.vertices:
            if point.x < 0 or point.y < 0 or point.x > self.format.x or point.y > self.format.y:
                self.error("The drawing extends outside the paper. Will not draw")
                return False
        if len(shape.vertices) > 0:
            if shape.ignore_ends:
                start = shape.vertices[1]
            else:
                start = shape.vertices[0]
            # move to start of shape
            if not self.move_to(start):
                return False

            if len(shape.vertices) == 1:
                self.pen_down()
                self.pen_up()
            else:
                # draw line from point to point in shape
                num_lines = len(shape.vertices) - 1
                if shape.ignore_ends:
                    num_lines -= 1
                for idx in range(num_lines):
                    # get all points on the edge from current point to the next one
                    points = shape.get_segment(idx, self.resolution)
                    # draw line from point to point, starting from the next one on the edge
                    for point in points[1:]:
                        if not self.line_to(point):
                            return False
        return True

    @do_action
    def draw_shapes(self, shapes: List[Shape]) -> bool:
        for shape in tqdm(shapes):
            if not self.draw_shape(shape):
                return False

        return self.move_home()

    @do_action
    def stop_motors(self) -> bool:
        self.axidraw.plot_setup()
        self.axidraw.options.mode = "align"
        self.axidraw.plot_run()

    def close(self) -> None:
        self.stop_motors()

        if self.is_connected:
            self.axidraw.disconnect()   

    def fit_to_paper(self, shapes: List[Shape], aspect_ratio: float, margin: float):
        #scale_xx is the scale in x if a point with 1 in absciss is mapped to the
        # edge of the paper minus the margin   
        scale_xx = self.format.x - 2 * margin
        #scale_xy is the scale in x if a point with 1 in ordinate is mapped to the
        # edge of the paper minus the margin
        scale_xy = (self.format.y - 2 * margin) * aspect_ratio
        # we keep the smallest of the 2 scales
        scale_X = min(scale_xx, scale_xy)
        scale_Y = scale_X / aspect_ratio

        # scale all points
        for shape in shapes:
            for point in shape.vertices:
                point.x = self.format.x / 2 + (point.x - 0.5) * scale_X
                point.y = self.format.y / 2 + (point.y - 0.5) * scale_Y

        return shapes  

def distSq_to_shape(shape: Shape, point: Point) -> float:
    """
    Returns the square of the distance to a shape's ends from a given point
    """
    return min(point.distSq(shape.vertices[0]), point.distSq(shape.vertices[1]))

def optimize_simple(shapes: List[Shape]) -> List[Shape]:
    """
    Simple "sort" of list of shape. Essentially, the next item in the returned list
    is the closest one to previous one, e.g.
    [shape(0), closest_in_remaining(n-1), closest_in_remaining(n-2), closest_in_remaining(n-3) ...].
    This can be refered to as a greedy Nearest Neighbour algorithm
    """
    buffer = [shapes[0]]
    shapes = shapes[1:]

    for _ in tqdm(range(len(shapes) - 1)):
        # find closest shape to last shape in buffer
        closest_shape = min(
            shapes, key=partial(distSq_to_shape, point=buffer[-1].vertices[-1]))
        
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

def json_to_shapes(json_file) -> Tuple[List[Shape], float]:
    # get either the shape list or the single shape as a list of shape
    shapes = json_file.get('shapes', [json_file])

    buffer = []
    for shape in shapes:
        new_shape = Shape(
            vertices=[Point(vertex['x'], vertex['y']) for vertex in shape['vertices']],
            is_polygonal=shape['isPolygonal']
        )
        buffer.append(new_shape)
    
    aspect_ratio = shapes[0]['canvas_width'] / shapes[0]['canvas_height']

    return buffer, aspect_ratio

def json_to_config(json_file) -> Dict:
    pass

def json_to_canvas_size(json_to_file) -> Point:
    pass

def process_canvas_size_request(q: Queue, canvas_size):
    """
    Converts the results of a POST request into a usable config dict and adds it to the server queue
    """
    print('Setting canvas size.')
    point = json_to_canvas_size(canvas_size)
    q.put(point)

def process_draw_request(q: Queue, draw_request):
    """
    Converts the results of a POST request into a usable shapes and adds it to the server queue
    """
    print('Adding draw request to queue')
    shapes = json_to_shapes(draw_request)
    q.put(shapes)

def process_config_request(q: Queue, config_request):
    """
    Converts the results of a POST request into a usable config dict and adds it to the server queue
    """
    print('Applying following settings to axidraw:', config_request)
    json_to_config(config_request)
    q.put(config_request)

def draw_request(ax, data):
    """
    Handles drawing the specified shapes. Designed for use with the server approach
    """
    print('Drawing next set of requested shapes in queue.')
    if not ax.draw_shapes(shapes=data, ask_verification=True):
        print('Something went wrong during draw and the axidraw '
            'handler will now exit.')
        ax.close()
        exit(1)

def axidraw_runner(q: Queue) -> NoReturn:
    """
    Process runner for the axidraw server
    """
    print("Initializing Axidraw handler")
    ax = Axifresco()

    try:
        while 1:
            data = q.get()
            if isinstance(data, List[Shape]):
                draw_request(ax, data)
            elif isinstance(data, Point):
                ax.set_format(data)
            elif isinstance(data, Dict):
                ax.set_config(data)

            time.sleep(0.5)
    except KeyboardInterrupt:
        print('Shutting down axidraw before exiting')
        ax.move_home()
        try:
            ax.close()
        except:
            pass
        exit(0)

def get_model(model_name: str) -> int:
    """
    Convert argparse arguments into an actual usable model option
    """
    models = {
        'V3': 1,
        'SE/A3': 2,
        'XLX': 3,
        'MiniKit': 4,
    }
    return models[model_name]

def get_canvas_size(size):
    """
    Convert argparse arguments into an actual usable format
    """
    if type(size, List[str]):
        if len(size) != 2:
            raise Exception(
                'If specifying a custom paper size, specify it as [width] [length]')
        return Point(size[0], size[1])
    else:
        sizes = {
            'a3': FORMATS.A3,
            'a4': FORMATS.A4,
            'a5': FORMATS.A5,
            'A3': FORMATS.A3,
            'A4': FORMATS.A4,
            'A5': FORMATS.A5,
        }
        return sizes[size]

def args_to_config(args) -> Dict:
    """
    Converts arguments from an Argparser into a usable config
    """
    config = {}

    for option in OPTIONS:
        if option not in ignored:
            opt = getattr(args, option, None)
            if opt is not None:
                config[option] = opt
    print('Axidraw config:', config)
    return config

def draw_from_json(args: argparse.Namespace, filename: str, ax: Axifresco) -> None:
    # load file
    print(f'Loading file {filename}')
    with open(filename, 'r') as f:
        shapes = json.load(f)

    # convert to shape and fit to paper
    print('Processing json file')
    shapes, aspect_ratio = json_to_shapes(shapes)

    print('Expanding shape to fit the paper')
    shapes = ax.fit_to_paper(shapes, aspect_ratio, args.margin)

    # optimize path
    if args.optimize:
        print('Optimizing drawing path')
        shapes = optimize_simple(shapes)

    if args.preview:
        img = Image.new('RGB', (ax.format.x, ax.format.y), (0, 0, 0))
        for shape in shapes:
            img = shape.preview(img, scale=1, center=False)
        plt.imshow(img)
        print('If the displayed image does not correspond to what is expected, '
              'simply press ctrl-c in the terminal. Otherwise, close the '
            'preview window and continue normally.')
        plt.show()
        img.close()

    # draw
    print('Drawing...')
    ax.draw_shapes(shapes, ask_verification=True, allow_pause=True, go_home=True)

def test(ax: Axifresco) -> None:
    """
    Move the pen to the center of the canvas and then back to home
    """
    center = ax.format / 2

    try:
        ax.move_to(center, ask_verification=True)
        time.sleep(0.5)
        ax.move_home()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Loads a json file describing some Fresco shapes and prints them on an Axidraw')
    file_parser = parser.add_argument_group('file options')
    file_parser.add_argument('--filename', type=str, required=True, help='Path to file to draw or directory. '
                             'If a directory is specified, all the files in the directory will be drawn as '
                             'separate layers, in alphabetical order')
    file_parser.add_argument('--paper-size', type=get_canvas_size, default=FORMATS.A3, nargs='+',
                             help='Paper size. Specify either a3, a4, a5, A3, A4, A5, '
                             'or a custom size in mm, e.g. 209 458 for a paper of 209mm '
                             'wide by 458mm long')
    file_parser.add_argument('--optimize', action='store_true',
                             help='Enables path optimization to try and minimize the '
                             'amount of time the pen will be moved in the air')
    file_parser.add_argument('--margin', type=int, default=0, help='Enforces a margin (in mm) '
                             'on the paper. Because the drawing will always be scaled to occupy '
                             'all available space, this means the drawing will indeed be smaller.')
    file_parser.add_argument('--resolution', type=int, default=10,
                             help='When drawing spline based shapes, defines the number of '
                             'subdivisions to apply to the shape')
    file_parser.add_argument('--preview', action='store_true', help='Preview the drawing')
    axidraw_parser = parser.add_argument_group('axidraw options')
    axidraw_parser.add_argument('--speed-pendown', type=int,
                                help='Maximum XY speed when the pen is down (plotting). (1-100)')
    axidraw_parser.add_argument('--speed-penup', type=int, help='Maximum XY speed when the pen is up. (1-100)')
    axidraw_parser.add_argument('--accel', type=int, help='Relative acceleration/deceleration speed. '
                                'This will be ignored for non polygonal shapes(1-100)')
    axidraw_parser.add_argument('--pen-pos-down', type=int,
                                help='Pen height when the pen is down (plotting). (0-100)')
    axidraw_parser.add_argument('--pen-pos-up', help='Pen height when the pen is up. (0-100)')
    axidraw_parser.add_argument('--pen-rate-lower', help='Speed of lowering the pen-lift motor. (1-100)')
    axidraw_parser.add_argument('--pen-rate-raise', help='Speed of raising the pen-lift motor. (1-100)')
    axidraw_parser.add_argument('--pen-delay-down', help='Added delay after lowering pen. (ms)')
    axidraw_parser.add_argument('--pen-delay-up', help='Added delay after raising pen. (ms)')
    axidraw_parser.add_argument('--model', type=get_model, default=2, choices=['V3', 'SE/A3', 'XLX', 'MiniKit'],
                                help='Select model of AxiDraw hardware.')
    axidraw_parser.add_argument('--port', help='Specify a USB port or AxiDraw to use.')
    axidraw_parser.add_argument('--port-config', type=int,
                                help='Override how the USB ports are located. (0-2)')
    axidraw_parser.add_argument('--test', action='store_true', help='Test which will move the pen to '
                                'the center of the canvas and then back home')
    axidraw_parser.add_argument('--reset', action='store_true', help='Simply lifts the penup and switches off the motors')

    args = parser.parse_args()
    config = args_to_config(args)

    # init axidraw
    print('Initialisizing Axidraw...')
    try:
        ax = Axifresco(config=config, reset=args.reset, resolution=args.resolution)
        
        if args.reset:
            ax.stop_motors()
            ax.axidraw.disconnect()
        elif args.test:
            test(ax)
        else:
            if os.path.isdir(args.filename):
                files = os.listdir(args.filename)
                files = natsorted(files)
                files = [args.filename + '/' + file for file in files]
            else:
                files = [args.filename]

            for file in files:
                draw_from_json(args, file, ax)

        try:
            ax.close()
        except:
            print('Failed to close connection properly.')

        print('All done')
    except Exception as e:
        print('An exception occured:', e)
        print('Closing connection first before exiting...')
        try:
            ax.close()
        except:
            print('Failed to close connection properly.')
