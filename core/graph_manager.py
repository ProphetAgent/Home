import json
from py2neo import Graph, Node, Relationship
import configparser


def md5(input_str):
    import hashlib
    return hashlib.md5(input_str.encode('utf-8')).hexdigest()


class GraphManager:
    def __init__(self, name):
        # 读取配置文件
        config = configparser.ConfigParser()
        config.read('../config/config.ini')

        # 从配置文件中获取Neo4j的连接信息
        uri = config.get('neo4j', 'uri')
        user = config.get('neo4j', 'user')
        password = config.get('neo4j', 'password')
        # 连接到图形数据库
        self.graph = Graph(uri, user=user, password=password, name=name)

    def update_node_properties(self, node_type, hash_id, properties):
        """
        根据hash_id更新节点属性
        :param node_type: 节点类型，可以是'scene'或'action'
        """
        # 根据node_type设置标签
        if node_type not in ["Scene", "Action"]:
            raise ValueError("Invalid node_type. Must be 'scene' or 'action'.")

        # 编写Cypher查询，匹配节点并更新属性
        set_clauses = ', '.join([f"n.{key} = ${key}" for key in properties.keys()])

        query = f"""
        MATCH (n:{node_type} {{hash_id: $hash_id}})
        SET {set_clauses}
        RETURN n
        """
        # 执行查询并传入参数
        result = self.graph.run(query, hash_id=hash_id, **properties).data()

        if result:
            print(f"{node_type.capitalize()} node with hash_id {hash_id} has been updated.")
        else:
            print(f"No {node_type} node found with hash_id {hash_id}.")

    def delete_relationship(self, start_hash_id, end_hash_id):
        query = """
          MATCH (a:Action {hash_id: $start_hash_id})-[r]->(s:Scene {hash_id: $end_hash_id})
          DELETE r
          """
        self.graph.run(query, start_hash_id=start_hash_id, end_hash_id=end_hash_id)

    def delete_scene_and_relationships_by_hash_id(self, scene_hash_id):
        """
        根据指定的scene_hash_id删除scene节点及其所有关系
        :param scene_hash_id: 场景的hash_id
        :return: 删除的scene节点及其所有关系
        """
        # 编写Cypher查询，查找并删除指定hash_id的scene节点及其所有关系
        query = """
        MATCH (s:Scene {hash_id: $scene_hash_id})
        DETACH DELETE s
        """

        # 执行查询，传递scene的hash_id作为参数
        self.graph.run(query, scene_hash_id=scene_hash_id)

        print(f"Scene with hash_id {scene_hash_id} and all related relationships have been deleted.")

    def get_scene_by_action_hash_id(self, action_hash_id):
        # 编写Cypher查询，查找从指定action节点指向的scene节点
        query = """
        MATCH (a:Action {hash_id: $action_hash_id})-[:LEADS_TO]->(s:Scene)
        RETURN s
        """

        # 执行查询，传递参数
        result = self.graph.run(query, action_hash_id=action_hash_id).data()
        # 如果找到匹配的scene节点，提取并返回scene的属性
        if result:
            print(result)
            scene = result[0]['s']
            return {
                "elementId": scene.get('elementId'),
                "id": scene.get('id'),
                "description": scene.get('description'),
                "hash_id": scene.get('hash_id'),
                "name": scene.get('name')
            }
        else:
            return None

    def get_outgoing_actions(self, scene_hash_id):
        """
        获取指定场景的出度动作
        :param scene_hash_id: 场景的hash_id
        :return: 出度动作列表
        """
        # 编写查询语句
        query = """
                    MATCH (s:Scene {hash_id: $hash_id})-[r:LEADS_TO]->(a:Action)
                    RETURN a.hash_id AS hash_id, a.name AS name, a.description AS description, a.bounds AS bounds, 
                           a.resource_id AS resource_id, a.element_semantic AS element_semantic
                    """
        # 执行查询
        results = self.graph.run(query, hash_id=scene_hash_id)

        # 收集所有结果
        actions = []
        for record in results:
            # 反序列化bounds
            bounds = json.loads(record['bounds'])
            actions.append({
                "hash_id": record['hash_id'],
                "name": record["name"],
                "element_semantic": record["element_semantic"],
                "description": record["description"],
                "bounds": bounds,
                "resource_id": record["resource_id"],
            })
        return actions

    def get_info_from_scene_list(self, scene_list):
        """
        根据场景列表获取场景信息
        :param scene_list: 场景列表
        :return: 场景信息列表
        """
        # List to hold the results
        scene_info_list = []

        # Query template to retrieve scene information
        query_template = """
        MATCH (scene:Scene {hash_id: $hash_id})
        RETURN scene.hash_id AS hash_id, scene.name AS name, scene.description AS description
        """
        try:
            # Iterate over each scene hash in the list
            for scene_hash in scene_list:
                # Run the query with the current scene hash
                result = self.graph.run(query_template, hash_id=scene_hash)
                # Fetch all records
                for record in result:
                    # Append each scene's info to the list as a dictionary
                    scene_info_list.append({
                        "hash_id": scene_hash,
                        "name": record["name"],
                        "description": record["description"]
                    })
            return scene_info_list
        except Exception as e:
            print("An error occurred:", e)
            return []

    def get_info_from_action_list(self, action_list):
        """
        根据动作列表获取动作信息
        :param action_list: 动作列表
        :return: 动作信息列表
        """
        # List to hold the results
        action_info_list = []

        # Query template to retrieve action information
        query_template = """
        MATCH (action:Action {hash_id: $hash_id})
        RETURN action.hash_id AS hash_id, action.name AS name, action.description AS description,
               action.element_semantic AS element_semantic, action.bounds AS bounds, action.resource_id AS resource_id
        """

        # Iterate over each action hash in the list
        for action_hash in action_list:
            # Run the query with the current action hash
            result = self.graph.run(query_template, hash_id=action_hash)
            # Fetch all records
            for record in result:
                # Append each action's info to the list as a dictionary
                action_info_list.append({
                    "hash_id": record["hash_id"],
                    "name": record["name"],
                    "description": record["description"],
                    "element_semantic": record["element_semantic"],
                    "bounds": record["bounds"],
                    "resource_id": record["resource_id"]
                })

        return action_info_list

    def build_graph_node(self, node_info):
        """
        根据节点信息构建节点
        """
        # 根据start_state, action_state, stop_state在neo4j中构图，代表一个场景通过一个动作到达下一个场景，会传入很多个这样的view_info
        # 创建或获取开始场景节点
        start_hash = node_info['start_state']['hash_id']
        end_hash = node_info['stop_state']['hash_id']
        action_str = f"{start_hash}-{end_hash}-{node_info['action_state']['resource_id']}-{node_info['action_state']['event_type']}"
        action_hash = md5(action_str)
        start_node = Node("Scene", hash_id=node_info['start_state']['hash_id'],
                          name=node_info['start_state']['name'],
                          description=node_info['start_state']['description'])
        self.graph.merge(start_node, "Scene", "hash_id")

        # 创建或获取结束场景节点
        stop_node = Node("Scene", hash_id=node_info['stop_state']['hash_id'],
                         name=node_info['stop_state']['name'],
                         description=node_info['stop_state']['description'])
        self.graph.merge(stop_node, "Scene", "hash_id")

        # 创建或获取动作节点
        action_node = Node("Action", hash_id=action_hash,
                           name=node_info['action_state']['name'],
                           element_semantic=node_info['action_state']['element_semantic'],
                           description=node_info['action_state']['description'],
                           bounds=json.dumps(node_info['action_state']['bounds']),
                           resource_id=node_info['action_state']['resource_id'],
                           event_type=node_info['action_state']['event_type']
                           )
        self.graph.merge(action_node, "Action", "hash_id")

        # 创建关系：开始场景到动作
        start_to_action_rel = Relationship(start_node, "LEADS_TO", action_node)
        self.graph.merge(start_to_action_rel)

        # 创建关系：动作到结束场景
        action_to_stop_rel = Relationship(action_node, "LEADS_TO", stop_node)
        self.graph.merge(action_to_stop_rel)


if __name__ == '__main__':
    pass
