import datetime
import json
import os
from natsort import natsorted
import xml.etree.ElementTree as ET
from colorama import Fore
from xml_extract import UIXMLTree
from tree_node import build_tree_from_xml
from core.azure_gpt4 import ask_gpt4o
from core.embedding import embeddings
from core.graph_manager import GraphManager
from agent_semantic.prompts.semantic_prompt import *

figure_dict = {}
language = "english"


class SemanticAgent:
    def __init__(self):
        pass

    def get_system_prompt(self, language, type):
        if language == "english":
            if type == "click":
                return SYSTEM_PROMPT_CLICK_ENGLISH_V1 + example_scene + example_action
            else:
                return SYSTEM_PROMPT_SWIPE_ENGLISH_V1 + example_scene + example_swipe

    def compress_xml_to_one_line(self, xml_text):
        from xml.dom import minidom
        # 解析 XML 字符串
        root = ET.fromstring(xml_text)
        # 将 ElementTree 转换为字符串
        rough_string = ET.tostring(root, 'utf-8')
        # 使用 minidom 进行格式化
        reparsed = minidom.parseString(rough_string)
        # 将格式化后的 XML 压缩成一行
        one_line_xml = reparsed.toxml()
        # 移除换行和多余空格
        one_line_xml = "".join(line.strip() for line in one_line_xml.split("\n"))
        return one_line_xml

    def check_webKit(self, xml_data):
        # 解析XML字符串
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError:
            return False
        # 使用XPath查找所有具有特定 base-class 属性的节点
        webview_nodes = root.findall(".//node[@base-class='android.webkit.WebView']")

        # 检查是否找到了WebView
        if webview_nodes:
            print("找到含有 WebView 的 base-class 字段。")
            return True
        else:
            return False

    def select_target_root_node(self, xml):
        exlude_package = ["com.android.systemui", "com.github.uiautomator"]
        # iterate all the root nodes from the xml node, and select the one we want
        root = ET.fromstring(xml)
        for child in root:
            if child.tag == "node" and child.get("package") in exlude_package:
                continue
            return ET.tostring(child, encoding='unicode')  # 或者使用 encoding='utf-8' 根据需要
        return None

    def extract_text(self, filename):
        # 查找 "view_" 和 ".png" 的开始索引
        start_index = filename.find("view_")
        # real device use png
        end_index = filename.find(".png", start_index)

        if start_index != -1 and end_index != -1:
            # 调整start_index到"view_"之后的字符
            start_index += len("view_")
            # 提取并返回需要的部分
            return filename[start_index:end_index]
        else:
            return "No match found"

    def load_figure(self, path):
        views_path = os.path.join(path, "views")
        for pic in os.listdir(views_path):
            if pic.startswith("view_"):
                view_name = self.extract_text(pic)
                figure_dict[view_name] = os.path.join(views_path, pic)

    def execute_description(self, event_data, event_info, json_path):
        view_hash = event_info['view']['view_str']
        figure_path = ""
        if view_hash in figure_dict:
            figure_path = figure_dict[view_hash]
        if event_info['event_type'] == 'touch':
            action_type = 'CLICK'
        elif event_info['event_type'] == 'long_touch':
            action_type = 'LONG PRESS'
        else:
            action_type = event_info['event_type']
        action_node = event_info['view']
        action_node_info = ""
        given_key = ['visible', 'checkable', 'editable', 'clickable', 'is_password', 'focusable', 'enabled', 'content_description', 'focused', 'resource_id', 'checked', 'text', 'class', 'scrollable', 'selected']
        for key, value in action_node.items():
            if key in given_key:
                action_node_info += f"{key}: {value}\n"
        ui_xml_tree = UIXMLTree()
        xml_pre = event_data['start_xml']
        xml_after = event_data['stop_xml']
        try:
            xml_pre_reduced = ui_xml_tree.process(self.select_target_root_node(xml_pre), app_name="AnkiDroid", level=1, str_type="plain_text", remove_system_bar=True, use_bounds=False, merge_switch=False)
            xml_after_reduced = ui_xml_tree.process(self.select_target_root_node(xml_after), app_name="AnkiDroid", level=1, str_type="plain_text", remove_system_bar=True, use_bounds=False, merge_switch=False)
        except Exception as e:
            print(e, json_path)
            return -1
        # xml_pre_reduced = UIState(UITree(self.compress_xml_to_one_line(xml_pre), is_json=False)).to_xml()
        # xml_after_reduced = UIState(UITree(self.compress_xml_to_one_line(xml_after), is_json=False)).to_xml()

        tree_start = build_tree_from_xml(xml_pre)
        tree_stop = build_tree_from_xml(xml_after)
        activity_start = tree_start.get_activity_list()
        activity_stop = tree_stop.get_activity_list()
        fragment_start = tree_start.get_fragment_list()
        fragment_stop = tree_stop.get_fragment_list()

        # check webview, if so, use ocr
        ocr_string_list_pre, ocr_string_list_after = [], []
        webview_flag_pre, webview_flag_after = self.check_webKit(xml_pre), self.check_webKit(xml_after)
        if webview_flag_pre:  # 检查是否为webview界面
            print("page is webview, execute OCR")
        if webview_flag_after:
            print("page is webview, execute OCR")
        # if webview_flag_pre:
        #     ocr_string_list_pre = self.ocr_webview(tree_start)
        user_prompt = USER_PROMPT_ENGLISH_V1.format(action_type=action_type,
                                                    action_node=action_node_info,
                                                    activity_start=activity_start,
                                                    activity_stop=activity_stop,
                                                    fragment_start=fragment_start,
                                                    fragment_stop=fragment_stop,
                                                    xml_pre_reduced=xml_pre_reduced,
                                                    xml_after_reduced=xml_after_reduced,
                                                    )
        # print(user_prompt)
        if action_type == 'CLICK' or action_type == 'LONG_CLICK':
            system_prompt = self.get_system_prompt(language, type="click")
        else:
            system_prompt = self.get_system_prompt(language, type="swipe")

        # gpt_out = ask_gpt4_1106(system_prompt, user_prompt, need_json=True, language=language)
        gpt_out = ask_gpt4o(system_prompt, user_prompt, [figure_path])
        if gpt_out is None:
            return -1
        print(gpt_out)
        print(Fore.CYAN + "action_name:", gpt_out['action_name'])
        # print("action_description:", gpt_out['action_description'])
        print("element_semantic:", gpt_out['element_semantic'])
        print("previous_page_name:", gpt_out['previous_page_name'])
        print("current_page_name:", gpt_out['current_page_name'] + Fore.RESET)
        time_identifier = datetime.datetime.now().strftime("%m%d%H%M")
        event_data["gpt_out"] = {
            "time": time_identifier,
            "action_name": gpt_out['action_name'],
            "action_description": gpt_out['action_description'],
            "element_semantic": gpt_out['element_semantic'],
            "reason": gpt_out.get("reason", ""),
            "previous_page_name": gpt_out.get("previous_page_name", ""),
            "previous_page_description": gpt_out.get("previous_page_description", ""),
            "current_page_name": gpt_out.get("current_page_name", ""),
            "current_page_description": gpt_out.get("current_page_description", ""),
        }
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(event_data, file, ensure_ascii=False, indent=4)
        return 0

    def execute_embedding(self, event_data, json_path):
        gpt_out = event_data['gpt_out']
        word_list = [gpt_out['action_name'], gpt_out['element_semantic'], gpt_out['previous_page_name'], gpt_out['current_page_name']]
        embedding_list = embeddings(word_list)
        event_data["embedding"] = {
            "action_name_embedding": embedding_list[0],
            "element_semantic_embedding": embedding_list[1],
            "previous_page_name_embedding": embedding_list[2],
            "current_page_name_embedding": embedding_list[3],
        }
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(event_data, file, ensure_ascii=False, indent=4)
        return 0

    def load_dir_and_execute(self, path, method, graph_name=""):
        index, error_num = 0, 0
        events_path = os.path.join(path, 'events')
        for event in natsorted(os.listdir(events_path)):
            if event.endswith('.json'):
                json_path = os.path.join(events_path, event)
                with open(json_path, 'r') as f:
                    try:
                        # Some event files are empty, filter them out
                        event_data = json.load(f)
                    except Exception as e:
                        print(e, json_path)
                        continue
                    event_info = event_data['event']
                # get semantic for each scene and action
                if method == "semantic":
                    if "gpt_out" in event_data:
                        continue
                    if "view" in event_info:  # means it is a key event
                        res = self.execute_description(event_data, event_info, json_path)
                        if res == -1:
                            error_num += 1
                        else:
                            index += 1
                # get embedding for each semantic result
                elif method == "embedding":
                    if "gpt_out" not in event_data or "view" not in event_info:
                        continue
                    res = self.execute_embedding(event_data, json_path)
                    if res == -1:
                        error_num += 1
                    else:
                        index += 1
                    print(index)
                # build a graph using neo4j
                elif method == "build_graph":
                    graph_manager = GraphManager(graph_name)
                    if "gpt_out" not in event_data or "embedding" not in event_data or "view" not in event_info:
                        continue
                    node_info = {
                        "start_state": {
                            "hash_id": event_data['start_state'],
                            "name": event_data['gpt_out']['previous_page_name'],
                            "description": event_data['gpt_out']['previous_page_description'],
                        },
                        "action_state": {
                            "hash_id": event_info['view']['view_str'],
                            "name": event_data['gpt_out']['action_name'],
                            "element_semantic": event_data['gpt_out']['element_semantic'],
                            "description": event_data['gpt_out']['action_description'],
                            "bounds": event_info['view']['bounds'],
                            "resource_id": event_info['view']['resource_id'],
                            "event_type": event_info['event_type'],
                            "class": event_info['view']['class'],
                        },
                        "stop_state": {
                            "hash_id": event_data['stop_state'],
                            "name": event_data['gpt_out']['current_page_name'],
                            "description": event_data['gpt_out']['current_page_description'],
                        }
                    }
                    graph_manager.build_graph_node(node_info)
        print(f"index: {index}")


if __name__ == '__main__':
    '''Please create a graph database in neo4j and fill the config/config.ini before running'''
    semantic_agent = SemanticAgent()
    root_path = "/resources/output-anki"
    semantic_agent.load_figure(root_path)
    semantic_agent.load_dir_and_execute(root_path, "semantic")
    semantic_agent.load_dir_and_execute(root_path, "embedding")
    semantic_agent.load_dir_and_execute(root_path, "build_graph", "anki")
