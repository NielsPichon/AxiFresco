import os
import logging
import argparse

from natsort import natsorted


from axifresco.axifresco import (
    args_to_config,
    get_canvas_size,
    get_model,
    Axifresco,
    test_center,
    draw_from_json
)


def main():
    parser = argparse.ArgumentParser(
        'Loads a json file describing some Fresco shapes and prints them on an Axidraw')
    file_parser = parser.add_argument_group('file options')
    file_parser.add_argument('--filename', type=str, required=True, help='Path to file to draw or directory. '
                             'If a directory is specified, all the files in the directory will be drawn as '
                             'separate layers, in alphabetical order')
    file_parser.add_argument('--paper-size', type=str, default='a3', nargs='+',
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
    axidraw_parser.add_argument('--test-center', action='store_true', help='Test which will move the pen to '
                                'the center of the canvas and then back home')
    axidraw_parser.add_argument('--reset', action='store_true', help='Simply lifts the penup and switches off the motors')

    args = parser.parse_args()
    config = args_to_config(args)

    # init axidraw
    logging.info('Initialisizing Axidraw...')
    try:
        ax = Axifresco(config=config, reset=args.reset, resolution=args.resolution)

        args.paper_size = get_canvas_size(args.paper_size)
        ax.set_format(args.paper_size)


        if args.reset:
            ax.stop_motors()
            ax.axidraw.disconnect()
        elif args.test_center:
            test_center(ax)
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
            logging.error('Failed to close connection properly.')

        logging.error('All done')
    except Exception as e:
        logging.error('An exception occured:', e)
        logging.info('Closing connection first before exiting...')
        try:
            ax.close()
        except:
            logging.error('Failed to close connection properly.')


if __name__ == '__main__':
    main()
