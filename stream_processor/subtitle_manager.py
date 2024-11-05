# stream_processor/subtitle_manager.py

import zmq
import logging

class SubtitleManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.zmq_port = 5555
        self.zmq_address = f"tcp://127.0.0.1:{self.zmq_port}"
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind(self.zmq_address)
        self.logger.info(f"ZeroMQ PUB socket bound to {self.zmq_address}")

    def update_subtitle(self, text: str):
        command = f'settext {text}'
        self.zmq_socket.send_string(command)
        self.logger.info(f"Updated subtitle: {text}")

    def cleanup(self):
        self.zmq_socket.close()
        self.zmq_context.term()
