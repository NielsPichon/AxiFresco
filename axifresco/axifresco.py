import time
import json
import atexit
import logging
import argparse
from itertools import chain
from threading import Event
from functools import partial
from math import pi, sqrt, atan2
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union
from multiprocessing.connection import Connection

import colorama
from tqdm import tqdm
from colorama import Fore
from pynput import keyboard
from pyaxidraw import axidraw
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


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

class Status:
    PLAYING = 'playing'
    STOPPED = 'stopped'
    PAUSED = 'paused'

@dataclass
class Point:
    """A utility class storing a point with basic
    2D vector arithmetic. Units are in mm.
    """

    x: float = 0
    y: float = 0

    @classmethod
    def from_complex(cls, coords: complex):
        return cls(coords.real, coords.imag)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __add__(self, b: Union['Point', float]):
        if isinstance(b, Point):
            return Point(self.x + b.x, self.y + b.y)
        else:
            return Point(self.x + b, self.y + b)

    def __radd__(self, b:  Union['Point', float]):
        return self + b

    def __sub__(self, b:  Union['Point', float]):
        if isinstance(b, Point):
            return Point(self.x - b.x, self.y - b.y)
        else:
            return Point(self.x - b, self.y - b)

    def __rsub__(self, b:  Union['Point', float]):
        return -self + b

    def __mul__(self, b: Union['Point', float]):
        if isinstance(b, Point):
            return Point(self.x * b.x, self.y * b.y)
        else:
            return Point(self.x * b, self.y * b)

    def __rmul__(self, b:  Union['Point', float]):
        return self * b

    def __truediv__(self, b:  Union['Point', float]):
        if isinstance(b, Point):
            return Point(self.x / b.x, self.y / b.y)
        else:
            return Point(self.x / b, self.y / b)

    def __iter__(self):
        return [self.x, self.y]

    def __eq__(self, o: Any) -> bool:
        if isinstance(o, Point):
            return self.x == o.x and self.y == o.y
        else:
            raise Exception(f"Cannot compare {type(o)} and {Point}")

    def distSq(self, b: 'Point') -> float:
        if isinstance(b, Point):
            raise Exception('distSq can only be called to compute distance '
                            f'between Points, not Point and {type(b)}')
        direction = b - self
        return direction.x ** 2 + direction.y ** 2

    def norm(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y)

    def dot(self, b: 'Point') -> float:
        if isinstance(b, Point):
            raise Exception('dot can only be called to compute the dot product '
                            f'of 2 Points, not of a Point and a {type(b)}')
        return self.x * b.x + self.y * b.y

class FORMATS:
    A3 = Point(297, 420)
    A4 = Point(210, 297)
    A5 = Point(148, 210)


@dataclass
class Shape:
    vertices: List[Point]
    is_polygonal: bool
    ignore_ends: bool = False
    layer: int = 0

    def catmull_rom(self, p0: Point, p1: Point, p2: Point, p3: Point) -> List[float]:
        """Computes the 3rd order polynomial coefficients describing a
        Catmull Rom spline with alpha=0.5 passing through
        the 4 specified points"""
        a = 0.5 * (-p0 + 3 * p1 - 3 * p2 + p3)
        b = 0.5 * (2 * p0 - 5 * p1 + 4 * p2 - p3)
        c = 0.5 * (-p0 + p2)
        d = 0.5 * (2 * p1)
        return a, b, c, d

    def get_segment(self, idx: int, resolution: int = 10) -> List[Point]:
        """Returns a list of points describing the edge
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

    def preview(self, img: Image = None, scale: float = 1,
                center: bool = True, flipX: bool = False,
                flipY: bool = False) -> Image:
        """Draw the shape using PIL. If an img is provided as
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

            width, height = img.size
            if flipX:
                x = [width - xx for xx in x]
            if flipY:
                y = [height - yy for yy in y]

            if center:
                x = [width + (xx - (max(x) + min(x)) * 0.5) for xx in x]
                y = [height + (yy - (max(y) + min(y)) * 0.5) for yy in y]

            xy = []
            for xx, yy in zip(x, y):
                xy.append(xx)
                xy.append(yy)

            draw.line(xy, fill=(255, 255, 255), width=1)

        return img

    def has_low_angle(self, idx: int, threshold: float = pi / 8, smooth_trajectory: bool = True) -> bool:
        """Checks whether 2 consecutive segments are roughly aligned.
        This may then be used to optimize tracing speed for instance.
        In the case of a curve shape, this is considered to always be true.
        """
        if not self.is_polygonal:
            return True
        elif not smooth_trajectory:
            return False
        else:
            # get segment from the end of the edge starting at the specified index
            extra_points = self.get_segment(idx + 1)
            # if there is no such edge, return False
            if (len(extra_points) > 0):
                b, c = extra_points
                a, _ = self.get_segment(idx)

                # compute angle of each segment w.r.t. the x-axis
                first = b - a
                first_angle = atan2(first.y, first.x)
                second = c - b
                second_angle = atan2(second.y, second.x)
                # compare the absolute difference between the 2
                # angles to the threshold
                return abs(second_angle - first_angle) < threshold
            else:
                return False


class Axifresco:
    """The main class to handle communication with the axidraw
    """

    def __init__(self, config: Dict = None, reset=False, resolution: int = 10,
                 unsafe: bool = False, pause_event: Event = None,
                 abort_event: Event = None,
                 status_pipe: Connection = None) -> None:
        # set the spline resolution
        self.resolution = resolution

        # set the safety mode. Unsafe means the axidraw will not
        # ask for confirmation before doing things.
        self.unsafe = unsafe

        # initialise axidraw API
        self.axidraw = axidraw.AxiDraw()
        self.axidraw.interactive()

        # init pause toggle
        self.pause = pause_event if pause_event is not None else Event()
        #init abort toggle
        self.abort = abort_event if abort_event is not None else Event()

        # set configuration
        if config is None:
            config = {}

        self.set_config(config)

        # init axidraw state
        self.is_connected = False
        self.position = Point(0, 0)
        self.status = Status.STOPPED

        # set default fromat to A3
        self.format = FORMATS.A3

        if not reset:
            # make sure we disconnect before exiting
            atexit.register(self.close)

            if not self.unsafe:
                print(Fore.GREEN + 'Please move the axidraw to the home position '
                    'and press a key to continue...')
                input()

        # queue for updating the status in the UI
        self.status_pipe = status_pipe

    def set_format(self, format: Point) -> None:
        self.format = format

    def set_config(self, config: Dict) -> bool:
        """Allows setting the axidraw options
        """
        # Force units to be mm
        config['units'] = 2

        try:
            for key, value in config.items():
                if key in OPTIONS:
                    exec(f'self.axidraw.options.{key} = {value}')
            self.axidraw.update()
        except Exception as e:
            logging.error(e)
            return False

        return True

    def do_action(func):
        """Wrapper for any action to perform on the axidraw.Before performing the action,
        user approval will be requested if relevant and connection/disconnection to
        the axidraw will be performed accordingly.
        """
        def action(self, *args, **kwargs) -> bool:
            ask_verification = kwargs.pop('ask_verification', False)
            disconnect_on_end = kwargs.pop('disconnect_on_end', False)
            allow_pause = kwargs.pop('allow_pause', False)
            go_home = kwargs.pop('go_home', False)

            if ask_verification and not self.unsafe:
                print(Fore.GREEN + f'Press any key to proceed with {func.__name__}...')
                input()

            if self.connect():
                if allow_pause and not self.unsafe:
                    # create a thread which listens for key presses and pauses the
                    # draw process accordingly
                    def on_press(key, pause_event, abort_event):
                        if not abort_event.is_set():
                            if key == keyboard.Key.space:
                                if pause_event.is_set():
                                    logging.info('Resuming draw.')
                                    self.status = Status.PLAYING
                                    pause_event.clear()
                                else:
                                    pause_event.set()
                                    self.status = Status.PAUSED
                                    logging.info('Drawing is paused.')
                                    print('Press [space] to resume or [escape] to abort.')
                            if key == keyboard.Key.esc and pause_event.is_set():
                                abort_event.set()

                    key_thread = keyboard.Listener(
                        on_press=partial(on_press, pause_event=self.pause, abort_event=self.abort))

                    print('Pause the drawing at any point by pressing [SPACEBAR]')
                    key_thread.start()

                ret = func(self, *args, **kwargs)

                if allow_pause and not self.unsafe:
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
                logging.error(e)
                return False
        else:
            return True

    @do_action
    def move_home(self) -> bool:
        try:
            return self.move_to(Point(0, 0))
        except Exception as e:
            logging.error(e)
            return False

    @do_action
    def pen_up(self) -> bool:
        try:
            self.axidraw.penup()
        except Exception as e:
            logging.error(e)
            return False
        return True

    @do_action
    def pen_down(self) -> bool:
        try:
            self.axidraw.pendown()
        except Exception as e:
            logging.error(e)
            return False
        return True

    @do_action
    def move_to(self, point: Point) -> bool:
        if self.position != point:
            try:
                if not self.wait_for_resume():
                    return False
                self.axidraw.moveto(point.y, point.x)
                self.position = point
            except Exception as e:
                logging.error(e)
                return False
        return True

    @do_action
    def line_to(self, point: Point) -> bool:
        try:
            if not self.wait_for_resume():
                return False
            self.axidraw.lineto(point.y, point.x)
            self.position = point
            if self.pause.is_set():
                self.pen_up()
        except Exception as e:
            logging.error(e)
            return False
        return True

    def wait_for_resume(self) -> bool:
        while self.pause.is_set():
            self.status = Status.STOPPED
            if self.abort.is_set():
                logging.info('Aborting...')
                try:
                    self.pause.clear()
                    self.move_home()
                    return False
                except:
                    logging.error('Something went wrong when aborting')
            time.sleep(0.1)
        return True

    @do_action
    def draw_shape(self, shape: Shape, smooth_trajectory: bool = False) -> bool:
        # quickly go through all points and make sure are within bounds of the canvas
        for point in shape.vertices:
            if point.x < 0 or point.y < 0 or point.x > self.format.x or point.y > self.format.y:
                logging.error("The drawing extends outside the paper. Will not draw")
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
                    # use acceleration if a strong angle is to come.
                    # Otherwise maximise speed by not using any acceleration
                    self.set_config({'const_speed': shape.has_low_angle(idx, smooth_trajectory=smooth_trajectory)})

                    # get all points on the edge from current point to the next one
                    points = shape.get_segment(idx, self.resolution)
                    # draw line from point to point, starting from the next one on the edge
                    for point in points[1:]:
                        if not self.line_to(point):
                            return False
        return True

    @do_action
    def draw_shape_v2(self, shape: Shape) -> bool:
        if len(shape.vertices) < 4:
            self.draw_shape(shape)

        # quickly go through all points and make sure are within bounds of the canvas
        for point in shape.vertices:
            if point.x < 0 or point.y < 0 or point.x > self.format.x or point.y > self.format.y:
                logging.error("The drawing extends outside the paper. Will not draw")
                return False

        if len(shape.vertices) > 0:
            start = 1 if shape.ignore_ends else 0

            # move to start of shape
            if not self.move_to(shape.vertices[start]):
                return False

            if len(shape.vertices) == 1:
                self.pen_down()
                self.pen_up()
            else:
                end = len(shape.vertices)
                if shape.ignore_ends:
                    end -= 1
                # convert shape to only lines
                vtx = list(chain([shape.get_segment(idx, self.resolution) for idx in range(start, end)]))

                # plan the speed for each vertex
                edges = [vtx[i + 1] - vtx[i] for i in range(len(vtx) - 1)]
                dists = [e.norm() for e in edges]
                edges_dir = [e / d for e, d in zip(edges, dists)]

                speeds = [0]

                # TODO retrieve max speedpendown and accel rate
                max_v = self.axidraw.options.speed_pendown
                accel_rate = self.axidraw.options.accel
                cornering = 10

                max_accel_t = max_v / accel_rate
                min_accel_dist = 0.5 * accel_rate * max_accel_t * max_accel_t

                delta = cornering / 5000  # Corner rounding/tolerance factor.

                # compute the velocity if the only limit is cornering and accel from previous vertex
                for i in range(1, len(vtx) - 1):
                    # distance from prev vertex to this one
                    d = dists[i - 1]
                    # speed at prev vertex
                    v_prev = speeds[i - 1]

                    # compute max vel at vertex
                    if d > min_accel_dist:
                        v_current = max_v
                    else:
                        v_current = min(max_v, sqrt(v_prev * v_prev + accel_rate * d))

                    # we modify the velocity based on the corner angle
                    cos_factor = - edges_dir[i - 1].dot(edges_dir)
                    root_factor = sqrt((1 - cos_factor) / 2)
                    denominator = 1 - root_factor
                    if denominator > 0.0001:
                        rfactor = (delta * root_factor) / denominator
                    else:
                        rfactor = 100000
                    v_current = min(v_current, sqrt(accel_rate * rfactor))
                    speeds.append(v_current)
                speeds.append(0)

                # backpropagate speeds to make sure we never exceed the max deceleration
                for i in reversed(range(1, len(vtx) - 1)):
                    # distance from prev vertex to this one
                    d = dists[i]
                    # speed at next vertex
                    v_next = speeds[i + 1]

                    if speeds[i] > v_next and d < min_accel_dist:
                        speeds[i] = min(speeds[i], sqrt(v_next * v_next - d * accel_rate))

                # draw
                for i in range(1, len(vtx)):
                    try:
                        self.axidraw.plot_seg_with_v(vtx[i].x, vtx[i].y, speeds[i - 1], speeds[i])
                    except:
                        return False
        return True

    @do_action
    def draw_shapes(self, shapes: List[Shape], smooth_trajectory: bool = False, use_v2: bool = True) -> bool:
        start_time = time.time()
        for i, shape in tqdm(enumerate(shapes)):
            if self.status_pipe is not None:
                self.status = Status.PLAYING
                self.status_pipe.send({
                    'state': self.status,
                    'message': f'Drawing {len(shapes)} shapes...',
                    'progress': int(i / len(shapes) * 100)
                })
            if use_v2:
                if not self.draw_shape_v2(shape):
                    return False
            else:
                if not self.draw_shape(shape, smooth_trajectory=smooth_trajectory):
                    return False

        ellapsed_time = int(time.time() - start_time)

        if not self.move_home():
            return False

        if self.status_pipe is not None:
            if ellapsed_time >= 60:
                minutes = ellapsed_time // 60
                seconds = ellapsed_time % 60
                if minutes >= 60:
                    hours = minutes // 60
                    minutes = minutes % 60
                    formated_time = f'{hours}h {minutes}min {seconds}s'
                else:
                    formated_time = f'{minutes}min {seconds}s'
            else:
                formated_time = f'{ellapsed_time}s'

            self.status = Status.STOPPED
            self.status_pipe.send({
                'state': self.status,
                'message': 'Drawing Completed in ' + formated_time,
                'progress': 100
            })

        return True

    @do_action
    def stop_motors(self) -> bool:
        self.axidraw.plot_setup()
        self.axidraw.options.mode = "align"
        self.axidraw.plot_run()
        self.status = Status.STOPPED

    def close(self) -> None:
        self.stop_motors()

        if self.is_connected:
            self.axidraw.disconnect()
            self.is_connected = False

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
    """Returns the square of the distance to a shape's
    ends from a given point
    """
    return min(point.distSq(shape.vertices[0]), point.distSq(shape.vertices[1]))

def optimize_simple(shapes: List[Shape]) -> List[Shape]:
    """Simple "sort" of list of shape. Essentially,
    the next item in the returned list
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

    if not isinstance(shapes, list):
        shapes = [shapes]

    buffer = []
    for shape in shapes:
        new_shape = Shape(
            vertices=[Point(vertex['x'], vertex['y']) for vertex in shape['vertices']],
            is_polygonal=shape['isPolygonal'], layer=shape.get('layer', 0)
        )
        buffer.append(new_shape)

    aspect_ratio = shapes[0]['canvas_width'] / shapes[0]['canvas_height']

    return buffer, aspect_ratio

def get_model(model_name: str) -> int:
    """Convert argparse arguments into an actual usable model option
    """
    models = {
        'V3': 1,
        'SE/A3': 2,
        'XLX': 3,
        'MiniKit': 4,
    }
    return models[model_name]

def get_canvas_size(size):
    """Convert argparse arguments into an actual usable format
    """
    if len(size) > 1:
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
        return sizes[size[0]]

def args_to_config(args) -> Dict:
    """Converts arguments from an Argparser into a usable config
    """
    config = {}

    for option in OPTIONS:
        if option not in ignored:
            opt = getattr(args, option, None)
            if opt is not None:
                config[option] = opt
    logging.info('Axidraw config:', config)
    return config

def draw(shapes: List[Shape], aspect_ratio: float, ax: Axifresco, margin: float,
         optimize: bool = True, smooth_trajectory: bool = False,
         preview: bool = False, use_v2: bool = False
) -> None:
    logging.info('Expanding shape to fit the paper')
    shapes = ax.fit_to_paper(shapes, aspect_ratio, margin)

    # optimize path
    if optimize:
        logging.info('Optimizing drawing path')
        shapes = optimize_simple(shapes)

    if preview:
        img = Image.new('RGB', (ax.format.x, ax.format.y), (0, 0, 0))
        for shape in shapes:
            img = shape.preview(img, scale=1, center=False, flipX=False, flipY=True)
        plt.imshow(img)
        print('If the displayed image does not correspond to what is expected, '
              'simply press ctrl-c in the terminal. Otherwise, close the '
            'preview window and continue normally.')
        plt.show()
        img.close()

    # draw
    logging.info('Drawing...')
    ax.draw_shapes(
        shapes,
        ask_verification=True,
        allow_pause=True,
        go_home=True,
        smooth_trajectory=smooth_trajectory,
        use_v2=use_v2
    )

def draw_from_json(args: argparse.Namespace, filename: str, ax: Axifresco) -> None:
    # load file
    logging.info(f'Loading file {filename}')
    with open(filename, 'r') as f:
        shapes = json.load(f)

    # convert to shape and fit to paper
    logging.info('Processing json file')
    shapes, aspect_ratio = json_to_shapes(shapes)

    draw(shapes, aspect_ratio, ax, args.margin, args.optimize, args.preview)


def test_center(ax: Axifresco) -> None:
    """Move the pen to the center of the canvas and then back to home
    """
    center = ax.format / 2

    try:
        ax.move_to(center, ask_verification=True)
        time.sleep(0.5)
        ax.move_home()
    except Exception as e:
        logging.error(e)
