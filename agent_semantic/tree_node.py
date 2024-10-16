import xmltodict
import re
from enum import Enum


class ActionType(Enum):
    ACTION_UNKNOWN = 0
    CRASH = 1
    FUZZ = 2
    START = 3
    RESTART = 4
    CLEAN_RESTART = 5
    NOP = 6
    ACTIVATE = 7
    BACK = 8
    FEED = 9
    CLICK = 10
    LONG_CLICK = 11
    SCROLL_TOP_DOWN = 12
    SCROLL_BOTTOM_UP = 13
    SCROLL_LEFT_RIGHT = 14
    SCROLL_RIGHT_LEFT = 15
    SCHEMA_EVENT = 16
    SHELL_EVENT = 17
    RANDOM = 18
    ACTION_DOWN = 19
    ACTION_MOVE = 20
    ACTION_UP = 21
    SCROLL = 22
    SEARCH = 23
    HOVER = 24
    PERMISSION_UPDATE = 25
    BACKRETURN = 26
    NETWORK = 27
    TWEAK_MOCK = 28
    INPUT = 29
    LOGIN_EVENT = 30
    GET_TWEAK_INFO = 31


class TreeNode:
    def __init__(self, node, parent=None):
        self.bounds = None
        if "@bounds" in node:
            self.bounds = get_flat_bounds(node["@bounds"])
        self.resource_id = node.get("@resource-id", "")
        if "zhangzhao" in self.resource_id:
            self.resource_id = ""
        self.class_name = node.get("@class", "")
        self.base_class_name = node.get("@base-class", "")
        self.text = node.get("@text", "")
        self.content_desc = node.get("@content-desc", "")
        self.hint = node.get("@hint", "")
        self.view_hash = node.get("@view_hash", "")
        self.alpha = node.get("@alpha", "")
        self.tag = node.get("@tag", "")
        self.page_xml = node.get("@tag-page_xml", "")
        if "/" in self.page_xml:
            self.page_xml = self.page_xml.split("/")[-1]
        self.page_fragment = node.get("@tag-page_name_fragment", "")
        if "." in self.page_fragment:
            self.page_fragment = self.page_fragment.split(".")[-1]

        self.clickable = node.get("@clickable", "").lower() == "true"
        self.long_clickable = node.get("@long-clickable", "").lower() == "true"
        self.scrollable = node.get("@scrollable", "").lower() == "true"
        self.enable = node.get("@enabled", "true").lower() == "true"
        self.checkable = node.get("@checked", "").lower() == "true"
        self.editable = node.get("@editable", "").lower() == "true"
        self.selected = node.get("@selected", "").lower() == "true"

        self.scroll_type = node.get("@scroll-type", "none")
        self.scroll_direction = node.get("@scroll-direction", "")
        self.important_for_a11y = node.get("@important-for-a11y", "")
        self.activity_name = node.get("@tag-page_name_activity", "")
        self.elem_id = node.get("@elem_id", "")

        self.parent = parent
        self.children = []
        self.short_resource_id = self.resource_id
        self.short_class_name = self.class_name
        self.short_base_class_name = self.base_class_name
        if "id/" in self.resource_id:
            self.short_resource_id = self.resource_id.split("id/")[-1]

        if "." in self.class_name:
            self.short_class_name = self.class_name.split(".")[-1]

        if "." in self.base_class_name:
            self.short_base_class_name = self.base_class_name.split(".")[-1]

    def __str__(self):
        return "[resource-id:{}][class:{}][text:{}][content-desc:{}]{}[chidren-num:{}]".format(self.resource_id,
                                                                                               self.class_name,
                                                                                               self.text,
                                                                                               self.content_desc,
                                                                                               self.bounds,
                                                                                               len(self.children))

    def to_short_string(self):
        resource_id = self.resource_id
        if self.class_name in self.resource_id:
            resource_id = ""
        return "[resource-id:{}][class:{}][text:{}][content-desc:{}][clickable:{}]]{}".format(resource_id,
                                                                                              self.class_name,
                                                                                              self.text,
                                                                                              self.content_desc,
                                                                                              self.clickable,
                                                                                              self.bounds
                                                                                              )

    def add_child(self, node):
        self.children.append(node)

    def dump(self, idx=0, clickable=False, long_clickable=False, scrollable=False, enable=False):
        print_self = True
        if clickable and not self.is_clickable():
            print_self = False
        elif long_clickable and not self.is_long_clickable():
            print_self = False
        elif scrollable and not self.is_scrollable():
            print_self = False
        elif enable and not self.is_enable():
            print_self = False

        if print_self:
            print("{}:{}".format(idx, self))
        for child in self.children:
            child.dump(idx + 1, clickable, long_clickable, scrollable, enable)

    def dump_from_root(self):
        parents = []
        node = self
        while node.parent is not None:
            parents.append(node.parent)
            node = node.parent
        i = -1
        while len(parents) > 0:
            print("{}:{}".format(i, parents.pop()))
            i -= 1
        self.dump()

    def short_dump(self, idx=0):
        result = ""
        for i in range(idx):
            result += "-"
        result += self.to_short_string() + "\n"
        for child in self.children:
            result += child.short_dump(idx + 1)
        return result

    def to_html(self, idx=0):
        result = ""
        node_html = self.to_html_node()
        child_html = ""
        child_idx = idx
        if node_html != "":
            child_idx += 1
        for child in self.children:
            child_html += child.to_html(child_idx)
        if node_html != "":
            for i in range(idx):
                result += " "
            result += node_html + "\n" + child_html
            return result
        elif child_html != "":
            return child_html

        return ""

    def to_html_node(self):
        # result = "<button fragment=@ layout=&>{}</button>"
        # if self.page_fragment == "":
        #     result = result.replace(" fragment=@", "")
        # else:
        #     result = result.replace("@", self.page_fragment)
        # if self.page_xml == "":
        #     result = result.replace(" layout=&", "")
        # else:
        #     result = result.replace("&", self.page_xml)
        if not self.is_valid():
            return ""
        result = "<p>{}</p>"
        if self.is_editable():
            result = "<input>{}</input>"
        elif self.is_selected():
            result = "<p selected>{}</p>"
        elif self.is_clickable() or self.is_long_clickable():
            result = "<button>{}</button>"
        elif self.is_scrollable():
            result = "<div>{}</div>"

        if self.content_desc != "":
            return result.format(self.content_desc)
        elif self.text != "":
            return result.format(self.text)
        elif self.tag != "":
            return result.format(self.tag)
        else:
            return ""

    def is_valid(self):
        return (self.bounds[0] + self.bounds[2]) > 0 and (self.bounds[1] + self.bounds[3]) > 0  # and self.alpha == "1"

    def is_clickable(self):
        if self.clickable or self.checkable:
            return True
        if self.tag != "" or self.content_desc != "" or self.text != "":
            all_nodes = self.get_all_nodes()
            for node in all_nodes:
                if node.clickable:
                    return False
            return True
        return False

    def is_long_clickable(self):
        return self.long_clickable

    def is_scrollable(self):
        return self.scrollable

    def is_enable(self):
        return self.enable

    def is_editable(self):
        return self.editable

    def is_selected(self):
        if self.selected:
            return True
        for node in self.get_all_nodes():
            if node.selected:
                return True
        return False

    def is_have_text(self):
        return self.text is not None

    def get_scroll_types(self):
        result = []
        if self.scroll_direction != "":
            directions = self.scroll_direction.split(" ")
            for direction in directions:
                if direction == "right":
                    result.append(ActionType.SCROLL_RIGHT_LEFT)
                if direction == "left":
                    result.append(ActionType.SCROLL_LEFT_RIGHT)
                if direction == "up":
                    result.append(ActionType.SCROLL_TOP_DOWN)
                if direction == "down":
                    result.append(ActionType.SCROLL_BOTTOM_UP)
            return result
        if self.scroll_type == "vertical":
            result.append(ActionType.SCROLL_TOP_DOWN)
            result.append(ActionType.SCROLL_BOTTOM_UP)
            return result

        if self.scroll_type == "horizontal":
            result.append(ActionType.SCROLL_RIGHT_LEFT)
            result.append(ActionType.SCROLL_LEFT_RIGHT)
            return result

        return [ActionType.SCROLL_TOP_DOWN, ActionType.SCROLL_BOTTOM_UP, ActionType.SCROLL_RIGHT_LEFT,
                ActionType.SCROLL_LEFT_RIGHT]

    def get_action_types(self):
        result = []
        if not self.is_enable():
            return result
        if not self.is_valid():
            return result
        if self.is_clickable():
            result.append(ActionType.CLICK)
        if self.is_long_clickable():
            result.append(ActionType.LONG_CLICK)
        if isinstance(self.class_name, str) and "lynx" in self.class_name:
            return result
        if self.is_scrollable():
            scroll_types = self.get_scroll_types()
            result += scroll_types
        return result

    def get_key_info(self, use_child_info=False):
        key_info = ""
        if use_child_info:
            node_list = self.get_all_nodes()
            key_info_set = set()
            # 从子控件中提取关键信息
            key_resource_id = ""
            for node in node_list:
                if key_resource_id == "":
                    if node.tag != "":
                        key_resource_id = node.tag
                    elif node.short_resource_id != "":
                        key_resource_id = node.short_resource_id
                if node.content_desc != "":
                    key_info_set.add(node.content_desc)
                elif node.text != "":
                    key_info_set.add(node.text)
            if len(key_info_set) == 0:
                key_info = key_resource_id
            else:
                for k in key_info_set:
                    key_info += k + " "
                key_info = key_info.strip()
        else:
            if self.hint != "":
                key_info = self.hint
            elif self.content_desc != "":
                key_info = self.content_desc
            elif self.text != "":
                key_info = self.text
            elif self.tag != "":
                key_info = self.tag
            else:
                key_info = self.short_resource_id

        if key_info == "" and self.parent is not None:
            key_info = self.parent.get_key_info()

        return key_info

    def get_all_nodes(self):
        node_list = [self]
        queue = [self]
        while len(queue) > 0:
            node = queue.pop(0)
            node_list += node.children
            queue += node.children
        return node_list

    def get_fragment_list(self):
        result = []
        if self.page_fragment != "":
            result.append(self.page_fragment)
        for child in self.children:
            result += child.get_fragment_list()
        return list(set(result))

    def get_fragment_dict(self):
        # 初始化一个空字典来存储结果
        result = {}
        # 如果当前对象有page_fragment，则在字典中增加或更新计数
        if self.page_fragment != "":
            if self.page_fragment in result:
                result[self.page_fragment] += 1
            else:
                result[self.page_fragment] = 1
        # 递归处理所有子对象
        for child in self.children:
            child_fragments = child.get_fragment_dict()
            # 合并子对象的结果到当前结果中，更新频次
            for key, value in child_fragments.items():
                if key in result:
                    result[key] += value
                else:
                    result[key] = value
        return result

    def get_layout_list(self):
        result = []
        if self.page_xml != "":
            result.append(self.page_xml)
        for child in self.children:
            result += child.get_layout_list()
        return list(set(result))

    def get_activity_list(self):
        result = []
        if self.activity_name != "":
            result.append(self.activity_name)
        for child in self.children:
            result += child.get_activity_list()
        return list(set(result))


def is_same_node(node1, node2):
    return node1.resource_id == node2.resource_id and \
        node1.class_name == node2.class_name and \
        node1.text == node2.text and \
        node1.content_desc == node2.content_desc and \
        node1.bounds == node2.bounds


def get_flat_bounds(bounds):
    lst = re.findall(r"-?\d+", bounds)
    bounds = [int(item) for item in lst]
    # bounds = "[{},{}][{},{}]".format(lst[0], lst[1], lst[2], lst[3])
    return bounds


def build_tree_from_xml(xml):
    try:
        root = xmltodict.parse(xml, encoding="utf-8")
        tree_node = parse_xml(root['node'])
    except Exception as e:
        print("parse xml error, error={}".format(e))
        return None

    return tree_node


def parse_xml(root, parent=None):
    tree_node = TreeNode(root, parent)
    if "node" not in root and "lynx-root" not in root and "lynx-node" not in root:
        return tree_node
    if "node" in root:
        node = root["node"]
        deal_node(node, tree_node)
    if "lynx-root" in root:
        node = root["lynx-root"]
        deal_node(node, tree_node)
    if "lynx-node" in root:
        node = root["lynx-node"]
        deal_node(node, tree_node)

    return tree_node


def deal_node(node, tree_node):
    if isinstance(node, dict):
        child_node = parse_xml(node, tree_node)
        child_node.parent = tree_node
        tree_node.add_child(child_node)
    elif isinstance(node, list):
        for child in node:
            child_node = parse_xml(child, tree_node)
            child_node.parent = tree_node
            tree_node.add_child(child_node)
