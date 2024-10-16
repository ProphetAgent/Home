import json
import time
import os

import numpy as np
from natsort import natsorted
from itertools import islice
from sklearn.metrics.pairwise import cosine_similarity
from core.graph_manager import GraphManager
from core.azure_gpt4 import ask_gpt4o
from core.embedding import embeddings
from core.utils import print_with_color
from agent_execute.prompts.execute_prompt import *

Scene_hash_dict, Action_hash_dict, ACTION_SCENE_PAIR = {}, {}, {}


class ExecuteAgent:
    def __init__(self):
        pass

    def get_node_info(self, index, type, matched_embedding='', candidates_index_list=[]):
        try:
            if type == 'scene':
                key, value = next(islice(Scene_hash_dict.items(), index, None))
                return {
                    "scene_hash": key,
                    "page_name": value['page_name'],
                    "page_description": value['page_description']
                }
            elif type == 'action':
                if len(candidates_index_list) == 0:
                    for key, value in Action_hash_dict.items():
                        if value['embedding'] == matched_embedding:
                            return {
                                "action_hash": key,
                                "action_name": value['action_name'],
                                "action_description": value['action_description']
                            }
                    return {
                        "action_hash": "No Match"
                    }
                else:
                    key, value = next(islice(Action_hash_dict.items(), candidates_index_list[index], None))
                    return {
                        "action_hash": key,
                        "action_name": value['action_name'],
                        "action_description": value['action_description']
                    }
        except StopIteration:
            return None  # or raise an exception, or handle the case as needed

    def get_candidate(self, type, action_hash_list=[]):
        candidates = []
        candidates_index_list = []
        num = 0
        if type == "scene":
            for key, value in Scene_hash_dict.items():
                candidates.append(value['embedding'])
        elif type == "action":
            if len(action_hash_list) != 0:
                for node in action_hash_list:
                    action_index = 0
                    for key, value in Action_hash_dict.items():
                        if key == node['hash_id']:
                            candidates.append(value["embedding"])
                            candidates_index_list.append(action_index)
                            break
                        action_index += 1
                if len(candidates) == 0:
                    print("给定的出边候选集没有在候选集中匹配上，选取全部候选集")
                    # 标注完的集合中没有对应的动作节点，那么把所有的候选集都加入
                    for key, value in Action_hash_dict.items():
                        candidates.append(value['embedding'])
            else:  # 当没有候选动作节点时
                for key, value in Action_hash_dict.items():
                    candidates.append(value['embedding'])
        return candidates, candidates_index_list

    def build_pre_data(self, path):
        start_time = time.time()
        num_scene, num_action = 0, 0
        events_path = os.path.join(path, "events")
        for event in natsorted(os.listdir(events_path)):
            if not event.endswith('.json'):
                continue
            json_path = os.path.join(events_path, event)
            with open(json_path, 'r') as f:
                try:
                    # Some event files are empty, filter them out
                    event_data = json.load(f)
                except Exception as e:
                    # print(e, json_path)
                    continue
                event_info = event_data['event']
            if "gpt_out" not in event_data or "view" not in event_info:
                continue
            start_state = event_data['start_state']
            stop_state = event_data['stop_state']
            view_str = event_info['view']['view_str']
            Scene_hash_dict[start_state] = {
                'page_name': event_data['gpt_out']['previous_page_name'],
                'page_description': event_data['gpt_out']['previous_page_description'],
                'embedding': event_data['embedding']['previous_page_name_embedding'],
            }
            Scene_hash_dict[stop_state] = {
                'page_name': event_data['gpt_out']['current_page_name'],
                'page_description': event_data['gpt_out']['current_page_description'],
                'embedding': event_data['embedding']['current_page_name_embedding']
            }
            Action_hash_dict[view_str] = {
                'action_name': event_data['gpt_out']['action_name'],
                'action_description': event_data['gpt_out']['action_description'],
                'bounds': event_data['event']['view']['bounds'],
                'embedding': event_data['embedding']['action_name_embedding'],
                'action_type': event_data['event']['event_type']
            }
            ACTION_SCENE_PAIR[view_str, stop_state] = Action_hash_dict[view_str]
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Load Hive Data Execution time: {execution_time} seconds")

    def generate_executable_code(self, item_path, matched_node_list):
        action_index = 0
        executable_code = []
        for matched_node in matched_node_list:
            if action_index == 0:
                # first node is scene
                action_index += 1
                continue
            action_index += 1
            if matched_node['hash_id'] != 'No match':
                if matched_node['hash_id'] == 'input':
                    executable_code.append({
                        'action_type': 'input',
                        'action_name': 'input',
                        'bounds': matched_node['input_text']
                    })
                else:
                    executable_code.append({
                        'action_type': Action_hash_dict[matched_node['hash_id']]['action_type'],
                        'action_name': matched_node['action_name'],
                        'action_description': matched_node['action_description'],
                        'bounds': Action_hash_dict[matched_node['hash_id']]['bounds']
                    })
        path = os.path.join(item_path, "flow_path.json")
        with open(path, 'r') as file:
            data = json.load(file)
        data['note'] = 'success'
        data["code"] = executable_code
        with open(path, 'w') as file:
            json.dump(data, file, indent=2)

    def get_full_case_v2(self, item_path):
        correct_node_list = []
        with open(os.path.join(item_path, "flow_path.json"), 'r', encoding='utf-8') as file:
            flow_file = file.read()
        flow_description = json.loads(flow_file)
        steps = flow_description['steps']
        pre_conditions = flow_description['pre_conditions']
        correct_node_list.append({
            'back_trace_num': 0,
            'type': 'scene',
            'selected_hash': [],
            'scene_name': pre_conditions if pre_conditions != '' else 'main page',
            'scene_hash': []  # 可能有多个，列表方式
        })
        res = pre_conditions if pre_conditions != '' else 'main page'
        for step in steps:
            res += '，'
            res += step['action_des']
            correct_node_list.append({
                'back_trace_num': 0,
                'type': 'action',
                'selected_hash': [],
                'action_name': step['action_des'],
                'action_hash': [],
                'bounds': ''
            })
        return res, correct_node_list

    def get_full_case(self, item_path):
        res = ''
        correct_node_list = []
        index = 0
        for item_case in sorted(os.listdir(item_path)):
            # 遍历case里的每个动作
            item_case_path = os.path.join(item_path, item_case)
            if os.path.isdir(item_case_path) and not item_case.startswith('.'):
                file_description = os.path.join(item_case_path, "description.json")
                with open(file_description, 'r', encoding='utf-8') as file:
                    description_tmp = file.read()
                description_json = json.loads(description_tmp)
                if index == 0:
                    if 'pre_condition' in description_json and description_json['pre_condition'] != '':
                        res += description_json['pre_condition']
                        correct_node_list.append({
                            'back_trace_num': 0,
                            'type': 'scene',
                            'selected_hash': [],
                            'scene_name': description_json['pre_condition'],
                            'scene_hash': [description_json['last_scene_hash']]  # 可能有多个，列表方式
                        })
                    else:
                        res += '首页'
                        correct_node_list.append({
                            'back_trace_num': 0,
                            'type': 'scene',
                            'selected_hash': [],
                            'scene_name': '首页',
                            'scene_hash': []
                        })
                    res += '，'
                    res += description_json['human_description']
                    correct_node_list.append({
                        'back_trace_num': 0,
                        'type': 'action',
                        'selected_hash': [],
                        'action_name': description_json['human_description'],
                        'action_hash': [description_json['action_hash']]
                    })
                    index += 1
                else:
                    if description_json['human_description'] != '':
                        res += '，'
                        res += description_json['human_description']
                        correct_node_list.append({
                            'back_trace_num': 0,
                            'type': 'action',
                            'selected_hash': [],
                            'action_name': description_json['human_description'],
                            'action_hash': [description_json['action_hash']]
                        })
                    else:
                        # 遇到空节点，直接返回，一条用例结束
                        break
                    index += 1
        return res, correct_node_list

    def get_semantic_by_scene_hash(self, last_action_hash, scene_hash):
        # if scene_hash not in SCENE_hash_list:
        #     return None
        # else:
        #     return SCENE_hash_list[scene_hash]
        return ACTION_SCENE_PAIR[(last_action_hash, scene_hash)]

    def generate_code_from_cases(self, test_cases_path):
        graph_manager = GraphManager("ankidroidgraph")
        candidate_number = 10
        # 定义不想包含的前缀
        prefixes = ['case-base', 'case-hard', 'case-finished']
        for item in sorted(os.listdir(test_cases_path)):
            item_path = os.path.join(test_cases_path, item)
            if not any(item.startswith(prefix) for prefix in prefixes) and os.path.isdir(item_path):
                '''Start testing a new case to obtain a complete text description of the case and 
                a list of all nodes to be matched, including the name and hash of each node'''
                start_time = time.time()
                full_case, correct_node_list = self.get_full_case_v2(item_path)
                print(full_case)
                node_index, matched_node_list, action_hash_list = 0, [], []
                # 过滤.开头的文件和不是文件夹的
                while node_index < len(correct_node_list):  # 有第一个场景节点
                    matching_node = correct_node_list[node_index]
                    if matching_node['type'] == 'scene':
                        candidates_embedding_list, _ = self.get_candidate("scene")
                        candidate_num = 15 if len(candidates_embedding_list) >= 15 else len(
                            candidates_embedding_list)
                        matching_word = matching_node['scene_name']
                    else:
                        last_matched_node = matched_node_list[len(matched_node_list) - 1]
                        if 'out_scene_hash' in last_matched_node:
                            out_scene_hash = last_matched_node['out_scene_hash']
                            action_hash_list = graph_manager.get_outgoing_actions(out_scene_hash)
                            print("action_arrival_scene_hash", out_scene_hash)
                        elif 'scene_hash' in last_matched_node:
                            out_scene_hash = last_matched_node['scene_hash']
                            action_hash_list = graph_manager.get_outgoing_actions(out_scene_hash)
                            print("action_arrival_scene_hash", out_scene_hash)
                        else:
                            action_hash_list = []
                        candidates_embedding_list, candidates_index_list = self.get_candidate("action", action_hash_list)
                        print("动作候选集数量: ", len(candidates_embedding_list))
                        candidate_num = candidate_number if len(candidates_embedding_list) >= candidate_number else len(
                            candidates_embedding_list)
                        matching_word = matching_node['action_name']
                    target = embeddings([matching_word])
                    similarity_scores = cosine_similarity(target, candidates_embedding_list)
                    # 找出前n个最高相似度的索引
                    top_same_indices = np.argsort(similarity_scores[0])[-candidate_num:][::-1]
                    if matching_node['type'] == 'scene':
                        print_with_color(f"-------------开始匹配场景节点: {matching_word}, node index: {node_index}, sum: {len(correct_node_list) - 1}-------------", "cyan")
                        candidates_list = []
                        scene_list = []
                        best_node = None
                        for idx in top_same_indices:
                            node_info = self.get_node_info(idx, 'scene')
                            node_info['similarity'] = similarity_scores[0][idx]
                            candidates_list.append(node_info)
                            scene_list.append(node_info['scene_hash'])
                            # print(f"索引：{idx}, 名称: {node_info['page_name']}, 相似度分数：{similarity_scores[0][idx]}, scene_hash: {node_info['scene_hash']}，描述：{node_info['page_description']}")
                        # print(candidates_list)
                        # 从图数据库中筛选并获取场景节点的信息
                        scene_info_list = graph_manager.get_info_from_scene_list(scene_list)
                        scene_info_prompt = ""
                        scene_index = 0
                        for node in scene_info_list:
                            for match in candidates_list:
                                if node['hash_id'] == match['scene_hash']:
                                    # 将图数据中的节点绑定上语义，给gpt进行选择
                                    node['page_name'] = match['page_name']
                                    node['page_description'] = match['page_description']
                                    node['similarity'] = match['similarity']
                                    scene_info_prompt += f"index of scene node candidate:{scene_index}, scene name: {node['page_name']}，scene description：{node['page_description']}，similarity：{node['similarity']}\n"
                                    scene_index += 1
                                    break
                            # print(node)
                        prompt = SCENE_PROMPT.format(matching_word=matching_word, candidate=scene_info_prompt, full_case=full_case)
                        print(prompt)
                        gpt_out = ask_gpt4o("", prompt, [], True)
                        print_with_color(str(gpt_out), "green")
                        gpt_choose_index = gpt_out['index']
                        choose_node_info = candidates_list[gpt_choose_index]
                        tmp_gpt_node = {
                            'hash_id': 'd64186fa379afbd442e52b82a3c0a02a',
                            'page_name': 'main page',
                            "page_description": 'This is the main page of the AnkiDroid app, displaying various elements such as the sidebar button, app title, card review status, and deck overview.',
                            'matching_word': matching_word,
                        }
                        choose_node_info['matching_word'] = matching_word
                        print_with_color(str(tmp_gpt_node), "green")
                        matched_node_list.append(tmp_gpt_node)
                        node_index += 1
                    else:
                        print_with_color(f"-------------开始匹配动作节点: {matching_word}, node index: {node_index}, sum: {len(correct_node_list) - 1}, 匹配次数：{matching_node['back_trace_num']}-------------", "cyan")
                        candidates_list = []
                        action_list = []
                        print(f"The {candidate_num} most matching vector indices and similarity scores")
                        for idx in top_same_indices:
                            node_info = self.get_node_info(idx, 'action', candidates_embedding_list[idx], candidates_index_list)
                            node_info['similarity'] = similarity_scores[0][idx]
                            candidates_list.append(node_info)
                            action_list.append(node_info['action_hash'])
                            # print(f"索引：{idx}, 名称: {node_info['action_name']}, 相似度分数：{similarity_scores[0][idx]}, action_hash: {node_info['action_hash']}，描述：{node_info['action_description']}")
                        # Filter and retrieve information on action nodes from the graph database
                        action_info_list = graph_manager.get_info_from_action_list(action_list)
                        send_gpt_action_content, gpt_index = "", 1
                        frequency_threshold = 0
                        filtered_action_info_list = []
                        for each_node in action_info_list:
                            if each_node['hash_id'] not in correct_node_list[node_index]['selected_hash']:
                                filtered_action_info_list.append(each_node)
                        action_info_list = filtered_action_info_list
                        for node in action_info_list:
                            tmp_gpt_node = f'index of action node candidate:{gpt_index}, '
                            for match in candidates_list:
                                # Combine the information obtained from the graph database into a new list that does not contain irrelevant information such as hash, add an index, and have GPT return the correct index
                                if node['hash_id'] == match['action_hash']:
                                    tmp_gpt_node += f'similarity with the matching text:{match["similarity"]}, action name:{match["action_name"]}, action description:{match["action_description"]}, action_hash:{match["action_hash"]}, '
                                    # 将图数据中的节点绑定上语义和相似度，给gpt进行选择
                                    node['action_name'] = match['action_name']
                                    node['action_description'] = match['action_description']
                                    node['similarity'] = match['similarity']
                                    # 对于候选集的所有动作节点，获取它对应到达的场景hash，再获取到达场景的语义，加入gpt，并加入整条case
                                    out_scene_info = graph_manager.get_scene_by_action_hash_id(node['hash_id'])
                                    if out_scene_info is None:
                                        continue
                                    out_scene_hash = out_scene_info['hash_id']
                                    # out_semantic = self.get_semantic_by_scene_hash(match['action_hash'], out_scene_hash)
                                    if out_scene_info:
                                        # print(f'action_hash: {node["hash_id"]}, 到达场景名：{out_scene_info["name"],}, 到达场景描述：{out_scene_info["description"]}')
                                        tmp_gpt_node += f'arrival scene name:{out_scene_info["name"]}, arrival scene description:{out_scene_info["description"]}\n'
                                        node['out_scene_hash'] = out_scene_hash
                                        node['arrival_scene_name'] = out_scene_info["name"]
                                        node['arrival_scene_description'] = out_scene_info["description"]
                                        gpt_index += 1
                                    break
                            send_gpt_action_content += tmp_gpt_node
                        prompt = ACTION_PROMPT.format(matching_word=matching_word, candidate=send_gpt_action_content, full_case=full_case)
                        print(prompt)
                        gpt_out = ask_gpt4o("", prompt, [], True)
                        print_with_color(str(gpt_out), "green")
                        if gpt_out is None:
                            print("gpt respond error, Set the action candidate set to all")
                            action_hash_list = []
                            matched_node_list.append({'action_hash': 'No match'})
                            node_index += 1
                            continue
                        gpt_choose_index = gpt_out['index'] - 1
                        if gpt_choose_index == -2:
                            # 输入操作
                            input_node = {
                                'hash_id': 'input',
                                'matching_word': matching_word,
                                'back_trace_num': correct_node_list[node_index]['back_trace_num'],
                                "action_name": "input",
                                "input_text": gpt_out['input_text'],
                            }
                            print_with_color(str(input_node), "green")
                            matched_node_list.append(input_node)
                            node_index += 1
                        elif gpt_choose_index != -1:
                            # Add the selected set of corresponding nodes to the hash provided by GPT, so that this hash will not be selected again during rollback
                            correct_node_list[node_index]['selected_hash'].append(action_info_list[gpt_choose_index]['hash_id'])
                            action_info_list[gpt_choose_index]['matching_word'] = matching_word
                            action_info_list[gpt_choose_index]['back_trace_num'] = correct_node_list[node_index]['back_trace_num']
                            matched_node_list.append(action_info_list[gpt_choose_index])
                            print_with_color(str(action_info_list[gpt_choose_index]), "green")
                            print_with_color(str(correct_node_list[node_index]), "blue")
                            node_index += 1
                        elif correct_node_list[node_index]['back_trace_num'] < 2:
                            # 可以让一个节点回退2次
                            if correct_node_list[node_index]['back_trace_num'] == 0:
                                if node_index > 0:
                                    print_with_color("Match failed, first rollback", "yellow")
                                    correct_node_list[node_index]['back_trace_num'] += 1
                                    node_index -= 1
                                    matched_node_list.pop()
                                else:
                                    # node_index 小于等于 0 的情况
                                    print_with_color("No Match, Set the action candidate set to all", "magenta")
                                    action_hash_list = []
                                    node_index += 1
                                    matched_node_list.append({'hash_id': 'No match', 'matching_word': matching_word, "back_trace_num": correct_node_list[node_index]['back_trace_num']})
                            elif correct_node_list[node_index]['back_trace_num'] == 1:
                                if node_index > 1:
                                    print_with_color("Match failed, second rollback", "yellow")
                                    correct_node_list[node_index]['back_trace_num'] += 1
                                    node_index -= 2
                                    matched_node_list.pop()
                                    matched_node_list.pop()
                                else:
                                    # node_index 为 1 或更小的情况
                                    print_with_color("No Match, Set the action candidate set to all", "magenta")
                                    action_hash_list = []
                                    node_index += 1
                                    matched_node_list.append({'hash_id': 'No match', 'matching_word': matching_word, "back_trace_num": correct_node_list[node_index]['back_trace_num']})
                            else:
                                print_with_color("No Match, Set the action candidate set to all", "magenta")
                                action_hash_list = []
                                node_index += 1
                                matched_node_list.append({'hash_id': 'No match', 'matching_word': matching_word, "back_trace_num": correct_node_list[node_index]['back_trace_num']})
                        else:
                            print_with_color("No Match, Set the action candidate set to all", "magenta")
                            action_hash_list = []
                            matched_node_list.append({'hash_id': 'No match', 'matching_word': matching_word, "back_trace_num": correct_node_list[node_index]['back_trace_num']})
                            node_index += 1
                print_with_color("----------------------------One test case execution completed----------------------------", "lightmagenta")
                print(full_case)
                print(matched_node_list)
                # generate and save executable_code to flow_path
                self.generate_executable_code(item_path, matched_node_list)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Generate Code time: {execution_time} seconds")
                break  # only generate one case for demonstration


if __name__ == '__main__':
    root_path = "../resources/output-anki"
    execute_agent = ExecuteAgent()
    execute_agent.build_pre_data(root_path)
    print(f"counts of scene:{len(Scene_hash_dict.keys())}, action:{len(Action_hash_dict.keys())}")
    execute_agent.generate_code_from_cases(os.path.join(root_path, "test_cases"))
