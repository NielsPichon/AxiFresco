from multiprocessing import Queue, Process
import json

from flask import Flask
from flask_restful import Api

from axifresco import process_canvas_size_request, \
                      process_draw_request, \
                      axidraw_runner


app = Flask(__name__)
api = Api(app)

@app.route('/draw', methods=['POST'])
def draw():
    print("Got a new draw request")
    draw_data = json.loads(request.data.decode())["input"]
    process_draw_request(q, draw_data)
    return "sent"


@app.route('/canvas_size', methods=['POST'])
def set_canvas_size():
    print("Got a new request for setting the canvas size")
    canvas_size = json.loads(request.data.decode())["input"]
    process_canvas_size_request(q, canvas_size)
    return "sent"

@app.route('/config', methods=['POST'])
def set_config():
    print("Got a new request for setting the config")
    config = json.loads(request.data.decode())["input"]
    process_config_request(q, config)
    return "sent"

if __name__ == '__main__':
    # Launch the axidraw manager
    q = Queue()
    axidraw = Process(target=axidraw_runner, args=(q,), daemon=True).start()

    # run the flask App
    app.run()