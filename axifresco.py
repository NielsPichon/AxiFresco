import inspect
import atexit
import time
from typing import Dict, NoReturn, List
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Queue

from tqdm import tqdm
from pyaxidraw import axidraw


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

@dataclass
class Point:
    """
    A utility class storing a point with basic 2D vector arithmetic. Units are in mm.
    """

    x: float = 0
    y: float = 0

    def __add__(self, b):
        self.x += b.x
        self.y += b.x

    def __sub__(self, b):
        self.x -= b.x
        self.y -= b.x
    
    def __iter__(self):
        return [self.x, self.y]

    def __eq__(self, o) -> bool:
        if isinstance(o, Point):
            return self.x == o.x and self.y == o.y
        else:
            raise Exception(f"Cannot compare {type(o)} and {Point}")


class FORMATS:
    A3 = Point(297, 420),
    A4 = Point(210, 297),
    A5 = Point(148, 210),



    @classmethod
    def do_action(action: Callable, *args, **kwargs) -> bool:
        """
        Wrapper for any action to perform on the axidraw. Before performaing the action,
        user approval will be requested if relevant and connection/disconnection to
        the axidraw will be performed accordingly.
        """
        ask_verification = kwargs.get('ask_verification', False)
        disconnect_on_end = kwargs.get('disconnect_on_end', False)
        
        if ask_verification:
            input(f'Press any key to proceed with {inspect.stack()[1].__name__}')

        if cls.connect():
            ret = action(*args, **kwargs)

            if disconnect_on_end:
                cls.disconnect()

            return ret
        else:
            return False


class Axifresco:
    """
    The main class to handle communication with the axidraw 
    """

    def __init__(self, config: Dict = None) -> NoReturn:
        # initialise axidraw API
        self.axidraw = axidraw.AxiDraw()
        self.axidraw.interactive()

        # set configuration
        if config is None:
            config = {}

        # Force units to be mm
        config['units'] = 2
        
        self.set_config(config)

        # init axidraw state
        self.is_connected = False
        self.position = Point(0, 0)

        # move the pen to the home position
        self.move_home(ask_verification=True)

        # set default fromat to A3
        self.format = FORMATS.A3

        # make sure we disconnect before exiting
        atexit.register(self.close)

    def set_format(self, format: Point) -> NoReturn:
        self.format = format

    def set_config(self, config: Dict) -> bool:
        """
        Allows setting the axidraw options
        """

        try:
            for key, value in config.items():
                if key in OPTIONS:
                    exec(f'self.axidraw.options.{key} = {value}')
        except Exception as e:
            print(e)
            return False

        return True
    
    def connect(self) -> bool:
        if not self.is_connected:
            try:
                self.axidraw.connect()
            except Exception as e:
                print(e)
                return False
        return True

    @do_action
    def go_home(self) -> bool:
        try:
            self.move_to(Point(0, 0))
        except Exception as e:
            print(e)
            return False

        return True

    @do_action
    def move_to(self, point: Point) -> bool:
        if self.position != point:
            try:
                self.axidraw.moveto(point.x, point.y)
                self.position = point
            except Exception as e:
                print(e)
                return False
        return True

    @do_action
    def line_to(self, point: Point) -> bool:
        try:
            self.axidraw.moveto(point.x, point.y)
            self.position = point
        except Exception as e:
            print(e)
            return False
        return True

    @do_action
    def draw_shape(self, points: List[Point]) -> bool:
        # quickly go through all points and make sure are within bounds of the canvas
        for point in points:
            if point.x < 0 or point.y < 0 or point.x > self.format.x or point.y > self.format.y:
                print("The drawing extends outside the paper. Will not draw")
                return False

        # move to start of shape
        if not self.moveto(points[0]):
            return False

        # draw line from point to point in shape
        for point in points[1:]:
            if not self.draw_line(self.position, point):
                return False

        return True

    @do_action
    def draw_shapes(self, shapes: List[List[Point]]) -> bool:
        for shape in tqdm(shapes):
            if not self.draw_shape(shape):
                return False
        return True


    def close(self):
        if self.is_connected:
            self.axidraw.disconnect()


def process_canvas_size_request(q: Queue, canvas_size):
    print(canvas_size)
    raise NotImplementedError
    x = 0
    y = 0
    q.put(Point(x, y))

def process_draw_request(q: Queue, draw_request):
    print(draw_request)
    raise NotImplementedError
    q.put()

def process_config_request(q: Queue, config_request):
    print(config_request)
    raise NotImplementedError

def draw(ax, data):
    if not ax.draw_shapes(shapes=data, ask_verification=True):
        print('Something went wrong during draw and the axidraw '
            'handler will now exit.')
        ax.close()
        exit(1)

def axidraw_runner(q: Queue) -> NoReturn:
    print("Initializing Axidraw handler")
    ax = Axifresco()

    try:
        while 1:
            data = q.get()
            if isinstance(data, List[List[Point]]):
                draw(ax, data)
            elif isinstance(data, Point):
                ax.set_format(data)
            elif isinstance(data, Dict):
                ax.set_config(data)

            time.sleep(0.5)
    except KeyboardInterrupt:
        print('Shutting down axidraw before exiting')
        ax.go_home()
        ax.close()
        exit(0)
