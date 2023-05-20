import os
import time
import webbrowser
from multiprocessing import Process


def run_tensorboard():
    os.system('tensorboard --logdir ./ray_result/')


if __name__ == "__main__":
    process: Process = Process(target=run_tensorboard)
    process.start()
    time.sleep(2)
    webbrowser.open('http://localhost:6006/')
