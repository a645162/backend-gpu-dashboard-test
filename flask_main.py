import json
import threading
from typing import List

from flask import Flask, request, Response
from flask_cors import CORS

from python_gpu_process import *

app = Flask(__name__)

# 允许所有域进行跨源请求
CORS(app)

global_gpu_info: List[dict] = []
global_gpu_usage: List[dict] = []
global_gpu_task: List[List[PythonGPUProcess]] = []


class GpuUsageThread(threading.Thread):
    def __init__(self, gpu_index: int):
        threading.Thread.__init__(self)
        self.gpu_index = gpu_index
        self.current_gpu_usage = {
            "gpuName": getRandomGpuName(),

            "coreUsage": 0,
            "memoryUsage": 0,

            "gpuMemoryTotalMB": 24.512 * 1024,

            "gpuMemoryTotal": "24",

            "gpuPowerUsage": 0,
            "gpuTDP": 450,
            "gpuTemperature": 0,
        }

    def run(self):
        while True:
            self.current_gpu_usage["coreUsage"] = getRandomNumber(0, 100)
            self.current_gpu_usage["memoryUsage"] = getRandomNumber(0, 100)

            self.current_gpu_usage["gpuPowerUsage"] = getRandomNumber(10, self.current_gpu_usage["gpuTDP"])
            self.current_gpu_usage["gpuTemperature"] = getRandomNumber(30, 80)

            global_gpu_usage[self.gpu_index].update(self.current_gpu_usage)

            time.sleep(2)


def generateGpu():
    global global_gpu_info
    global global_gpu_usage
    global global_gpu_task

    current_gpu_task_list: List[PythonGPUProcess] = []

    for _ in range(getRandomNumber(0, 2)):
        current_gpu_task_list.append(PythonGPUProcess())

    global_gpu_info.append({})
    global_gpu_usage.append({})
    global_gpu_task.append(current_gpu_task_list)

    new_thread = GpuUsageThread(len(global_gpu_info) - 1)
    new_thread.start()


def generateAllGpu(count: int):
    for _ in range(count):
        generateGpu()


@app.route("/get_system_info")
def get_system_info():
    system_info: dict = {
        "memoryPhysicTotalMb": 8192,
        "memoryPhysicUsedMb": 3000,

        "memorySwapTotalMb": 4096,
        "memorySwapUsedMb": 2048,
    }

    system_info["memoryPhysicUsedMb"] = \
        getRandomNumber(
            0,
            system_info["memoryPhysicTotalMb"]
        )
    system_info["memorySwapUsedMb"] = \
        getRandomNumber(
            0,
            system_info["memorySwapTotalMb"]
        )

    return Response(
        response=json.dumps(system_info),
        status=200,
        mimetype="application/json",
    )


@app.route("/get_gpu_count")
def get_gpu_count():
    return Response(
        response=json.dumps({"result": len(global_gpu_task)}),
        status=200,
        mimetype="application/json",
    )


@app.route("/get_gpu_usage")
def get_gpu_usage():
    gpu_index = (
        request.args.get(
            'gpu_index',
            default=None, type=int
        )
    )
    if gpu_index is None or gpu_index > len(global_gpu_usage):
        return Response(
            response=json.dumps({"result": "Invalid GPU Index(gpu_index)."}),
            status=400,
            mimetype="application/json",
        )

    current_gpu_info = global_gpu_info[gpu_index]
    current_gpu_usage = global_gpu_usage[gpu_index]

    response_gpu_usage = {
        "result": len(global_gpu_usage),

        "gpuName": "Test GPU",

        "coreUsage": 0,
        "memoryUsage": 0,

        "gpuMemoryUsage": "0",
        "gpuMemoryTotal": "0",

        "gpuPowerUsage": 0,
        "gpuTDP": 0,
        "gpuTemperature": 0,
    }

    response_gpu_usage.update(current_gpu_info)
    response_gpu_usage.update(current_gpu_usage)

    return Response(
        response=json.dumps(response_gpu_usage),
        status=200,
        mimetype="application/json",
    )


@app.route("/get_gpu_task")
def get_gpu_task():
    gpu_index = (
        request.args.get(
            'gpu_index',
            default=None, type=int
        )
    )

    if gpu_index is None or gpu_index > len(global_gpu_task):
        return Response(
            response=json.dumps({"result": "Invalid GPU Index(gpu_index)."}),
            status=400,
            mimetype="application/json",
        )

    current_gpu_processes: List[PythonGPUProcess] = global_gpu_task[gpu_index]

    task_list = []

    for process_obj in current_gpu_processes:
        task_list.append(
            {
                "id": process_obj.pid,
                "name": process_obj.user["name"],

                "debugMode": process_obj.is_debug,

                "projectName": process_obj.project_name,
                "pyFileName": process_obj.python_file,

                "runTime": process_obj.running_time_human,
                "startTimestamp": int(process_obj.start_time) * 1000,

                "gpuMemoryUsage": process_obj.task_gpu_memory >> 10 >> 10,

                "worldSize": process_obj.world_size,
                "localRank": process_obj.local_rank,
                "condaEnv": process_obj.conda_env,

                "command": process_obj.command,

                "taskMainMemoryMB": process_obj.task_main_memory_mb,
            }
        )

    response_gpu_tasks = {
        "result": len(task_list),
        "taskList": task_list
    }

    return Response(
        response=json.dumps(response_gpu_tasks),
        status=200,
        mimetype="application/json",
    )


flask_server_port = 8082


def start_web_server_ipv4():
    app.run(host="0.0.0.0", port=flask_server_port, debug=False)


def start_web_server_both():
    app.run(host="::", port=flask_server_port, threaded=True)


if __name__ == "__main__":
    generateAllGpu(8)

    start_web_server_ipv4()
    # start_web_server_both()
