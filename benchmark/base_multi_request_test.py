import subprocess
import json
import time

def start_subprocesses_concurrently(n):
    # curl 命令和参数
    url = "http://0.0.0.0:8080/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": "San Francisco is a",
        "max_tokens": 50,
        "temperature": 0
    }
    data_str = json.dumps(data)

    processes = []
    try:
        for i in range(n):
            # 构造 curl 命令
            command = [
                "curl", url,
                "-H", f"Content-Type: {headers['Content-Type']}",
                "-d", data_str
            ]
            # 启动子进程但不等待完成
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(process)
            print(f"第 {i + 1} 个子进程已启动")

        # 等待所有子进程完成
        results = []
        for i, process in enumerate(processes):
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                results.append(f"子进程 {i + 1} 输出: {stdout.decode().strip()}")
            else:
                results.append(f"子进程 {i + 1} 错误: {stderr.decode().strip()}")

        # 输出所有结果
        for result in results:
            print(result)

    except Exception as e:
        print(f"运行出错: {e}")

if __name__ == "__main__":
    n = 100
    runs = 5
    times = []

    for i in range(runs):
        print(f"开始第 {i + 1} 次测试...")
        start_time = time.time()
        start_subprocesses_concurrently(n)
        end_time = time.time()
        duration = end_time - start_time
        times.append(duration)
        print(f"第 {i + 1} 次完成，耗时: {duration:.2f} 秒")
        time.sleep(1)

    avg = sum(times) / len(times) if times else 0.0
    print(f"{runs} 次测试平均耗时: {avg:.2f} 秒")