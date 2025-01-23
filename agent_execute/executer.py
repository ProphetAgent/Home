import json
import os
import subprocess
import time

from core.utils import print_with_color
from androguard.core.bytecodes.apk import APK
def execute_adb(adb_command):
    # print(adb_command)
    result = subprocess.run(adb_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    print_with_color(f"Command execution failed: {adb_command}", "red")
    print_with_color(result.stderr, "red")
    return "ERROR"

def get_device_id():
    try:
        # 执行 adb devices 命令
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        # 解析输出
        lines = result.stdout.strip().split('\n')
        # 提取设备 ID（跳过第一行标题）
        if len(lines) > 1:
            device_info = lines[1].split()
            if len(device_info) > 0:
                return device_info[0]
    except Exception as e:
        print(f"Error: {e}")
    return None

class CommandExecuter:
    def __init__(self, device, apk_path, case_path):
        self.device = device
        self.apk_path = apk_path
        self.apk = APK(apk_path)
        self.package_name = self.apk.get_package()
        self.case_path = case_path
        self.width, self.height = self.get_device_size()

    def start_app(self):
        apk_path = self.apk_path
        package_name = self.package_name

        if not is_app_installed(self.device, package_name):
            print("App not installed. Installing...")
            install_app(self.device, apk_path)
        else:
            print("App already installed.")

        print("Launching app...")
        launch_app(self.device, package_name)

    def execute_case(self):
        for each_case_path in os.listdir(self.case_path):
            flow_path = os.path.join(self.case_path, each_case_path, "flow_path.json")
            with open(flow_path, 'r') as f:
                flow_file = json.load(f)
                if 'note' in flow_file and flow_file['note'] == 'success':
                    start_time = time.time()
                    print(f"Executing case: {flow_path}")
                    steps = flow_file['code']
                    for step in steps:
                        if step['action_type'] == 'touch':
                            self.tap(step['bounds'])
                        elif step['action_type'] == 'long_touch':
                            self.long_press(step['bounds'])
                        elif step['action_type'] == 'input':
                            self.text(step['bounds'])
                        time.sleep(1)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"Execution time: {execution_time} seconds")
    def calculate_center(self, bounds):
        if isinstance(bounds, list) and len(bounds) == 2:
            x_center = (bounds[0][0] + bounds[1][0]) / 2
            y_center = (bounds[0][1] + bounds[1][1]) / 2
            return x_center, y_center
        return None

    def get_device_size(self):
        adb_command = f"adb -s {self.device} shell wm size"
        result = execute_adb(adb_command)
        if result != "ERROR":
            return map(int, result.split(": ")[1].split("x"))
        return 0, 0

    def back(self):
        adb_command = f"adb -s {self.device} shell input keyevent KEYCODE_BACK"
        ret = execute_adb(adb_command)
        return ret

    def tap(self, bounds):
        x, y = self.calculate_center(bounds)
        adb_command = f"adb -s {self.device} shell input tap {x} {y}"
        ret = execute_adb(adb_command)
        return ret

    def text(self, input_str):
        input_str = input_str.replace(" ", "%s")
        input_str = input_str.replace("'", "")
        adb_command = f"adb -s {self.device} shell input text {input_str}"
        ret = execute_adb(adb_command)
        return ret

    def long_press(self, bounds, duration=1000):
        x, y = self.calculate_center(bounds)
        adb_command = f"adb -s {self.device} shell input swipe {x} {y} {x} {y} {duration}"
        ret = execute_adb(adb_command)
        return ret

    def swipe(self, x, y, direction, dist="medium", quick=False):
        unit_dist = int(self.width / 10)
        if dist == "long":
            unit_dist *= 4
        elif dist == "medium":
            unit_dist *= 3
        if direction == "up":
            offset = 0, -2 * unit_dist
        elif direction == "down":
            offset = 0, 2 * unit_dist
        elif direction == "left":
            offset = -1 * unit_dist, 0
        elif direction == "right":
            offset = unit_dist, 0
        else:
            return "ERROR"
        duration = 100 if quick else 400
        adb_command = f"adb -s {self.device} shell input swipe {x} {y} {x+offset[0]} {y+offset[1]} {duration}"
        ret = execute_adb(adb_command)
        return ret

    def swipe_precise(self, start, end, duration=400):
        start_x, start_y = start
        end_x, end_y = end
        adb_command = f"adb -s {self.device} shell input swipe {start_x} {start_x} {end_x} {end_y} {duration}"
        ret = execute_adb(adb_command)
        return ret
def is_app_installed(device_id, package_name):
    result = subprocess.run(
        ['adb', '-s', device_id, 'shell', 'pm', 'list', 'packages', package_name],
        capture_output=True,
        text=True
    )
    return package_name in result.stdout

def install_app(device_id, apk_path):
    subprocess.run(['adb', '-s', device_id, 'install', apk_path])

def launch_app(device_id, package_name):
    subprocess.run(['adb', '-s', device_id, 'shell', 'monkey', '-p', package_name, '-c', 'android.intent.category.LAUNCHER', '1'])


if __name__ == '__main__':
    device_id = get_device_id()
    if device_id:
        # please enter your apk and case path
        command_executer = CommandExecuter(device_id, '../apks/AnkiDroid-2.9.7.apk', '../resources/output-anki/test_cases')
        print(f"Connected device ID: {device_id}")
    else:
        print("No device found.")
        exit(1)
    command_executer.start_app()
    # wait for app start
    time.sleep(2)
    command_executer.execute_case()

