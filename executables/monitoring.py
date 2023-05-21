import os
import time
import webbrowser
from multiprocessing import Process


def run_tensorboard():
    os.system('tensorboard --logdir ./ray_result/')


if __name__ == "__main__":
    process: Process = Process(target=run_tensorboard)
    process.start()
    time.sleep(5)

    # Monitoring of learning
    url_monitoring_learning: str = 'http://localhost:6006/?pinnedCards=%5B%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22ray%2Ftune%2Fevaluation%2Fepisode_reward_mean%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22ray%2Ftune%2Fevaluation%2Fepisode_len_mean%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22ray%2Ftune%2Fepisode_reward_mean%22%7D%5D&darkMode=true#timeseries'
    webbrowser.open(url_monitoring_learning)

    # Monitoring simulation
    url_monitoring_simulation: str = 'http://localhost:6006/?pinnedCards=%5B%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22ray%2Ftune%2Fevaluation%2Fsampler_perf%2Fmean_env_render_ms%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22ray%2Ftune%2Fevaluation%2Fsampler_perf%2Fmean_action_processing_ms%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22ray%2Ftune%2Fevaluation%2Fsampler_perf%2Fmean_inference_ms%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22ray%2Ftune%2Fperf%2Fcpu_util_percent%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22ray%2Ftune%2Fperf%2Fgpu_util_percent0%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22ray%2Ftune%2Fperf%2Fram_util_percent%22%7D%5D&darkMode=true#timeseries'
    webbrowser.open(url_monitoring_simulation)
