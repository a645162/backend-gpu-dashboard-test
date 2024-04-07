from typing import List
import random
import time


def getRandomNumber(start, end):
    return random.randint(start, end)


def getRandomString(length: int) -> str:
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length))


def get_timestamp_before_seconds(sec):
    current_timestamp = time.time()
    timestamp_before = current_timestamp - sec
    return timestamp_before


def getRandomGpuName() -> str:
    list_name: List[str] = [
        "RTX 4090",
        "RTX 3090",
        "RTX 2080Ti",
        "GTX 1080Ti",
    ]
    current_name = list_name[getRandomNumber(0, len(list_name) - 1)]
    return current_name


def getRandomName() -> str:
    list_name: List[str] = [
        "孔昊旻",
        "Haomin",
        "Haomin Kong",
        "Kong",
        "KHM",
        "KM",
        "Unknown",
    ]
    current_name = list_name[getRandomNumber(0, len(list_name) - 1)]
    # current_name = current_name[:getRandomNumber(2, len(current_name))]
    return current_name


class PythonGPUProcess:

    def __init__(self):
        self.pid = getRandomNumber(10000, 100000)
        self.user = {"name": getRandomName()}
        self.is_debug = getRandomNumber(0, 3) == 0
        self.project_name = getRandomString(10)
        self.python_file = getRandomString(10) + ".py"
        self.running_time_human = ""
        self.world_size = getRandomNumber(1, 2)
        self.start_time = get_timestamp_before_seconds(
            getRandomNumber(0, 3600)
        )
        self.task_gpu_memory = int(12.5 * 1024 * 1024 * 1024)
        self.task_main_memory_mb = int(12.5 * 1024)

        self.local_rank = self.world_size - 1
        self.conda_env = "yolo"

        self.command = "python xxxx/train.py"


if __name__ == "__main__":
    # for i in range(10):
    #     print(getRandomNumber(0,2))
    print(get_timestamp_before_seconds(3600))
