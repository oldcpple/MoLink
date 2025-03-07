import subprocess
import json
import time
import threading

def send_request(index, url, headers, data):
    command = [
        "curl", url,
        "-H", f"Content-Type: {headers['Content-Type']}",
        "-d", json.dumps(data)
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print(f"请求 {index} 输出: {stdout.decode().strip()}")
        else:
            print(f"请求 {index} 错误: {stderr.decode().strip()}")
    except Exception as e:
        print(f"请求 {index} 运行出错: {e}")


def start_requests_concurrently(batch_size, interval):
    url = "http://localhost:8080/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": "San Francisco is a",
        "max_tokens": 30,
        "temperature": 0
    }
    request_count = 0
    
    try:
        while True:
            for i in range(batch_size):
                request_count += 1
                thread = threading.Thread(target=send_request, args=(request_count, url, headers, data))
                thread.start()
                print(f"第 {request_count} 个请求已提交")
            time.sleep(interval)  # 每秒提交 batch_size 个请求
    except KeyboardInterrupt:
        print("程序已停止")

if __name__ == "__main__":
    batch_size = int(input("请输入每次提交请求数量: "))
    interval = int(input("请输入每次提交请求的时间间隔: "))
    start_requests_concurrently(batch_size, interval)
