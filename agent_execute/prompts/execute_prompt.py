
SCENE_PROMPT = """
You are a smartphone assistant specifically crafted. 
Task: I will provide you with the scene text to be matched and scene candidate node information (given in JSON format). The candidate nodes are recorded from the random traversal of automated testing tools.
candidate node information includes
1. scene_index: The index of each scene nodes;
2. scene_name: The name of the scene node;
3. scene_description: The description of the scene node;
You need to select a node from the candidate nodes information I provide that best matches the text to be matched. When you feel that the information of these nodes is very similar, choose the node with the highest frequency and return it.
---
## The scene text to be matched from candidate node
{matching_word}
## candidate nodes
{candidate}
## Format example
{{
    "index": number // It should be of type Number, which is one of the candidate sets I provided. 
}}
---
Your response must be in JSON format, that carefully referenced "Format example".
"""
ACTION_PROMPT = """
You are a smartphone assistant specifically crafted. 
Matching task: There is a complete test case consisting of a starting scene description and multiple action texts. 
There is one action text in action texts, and you need to select a node from the given candidate set that best fits the action text.
The candidate set nodes are recorded from the random traversal of automated testing tools. 
Candidate node information includes 
1. action_index: The index of each action nodes; 
2. action_name: The name of the action node;
3. resource id: The resource_id of the operation control, may be empty;
4. arrival scene name: The scene name reached through this action;
5. arrival scene description: The scene description reached through this action; 7. index of action node candidate 
You need to select a node from the candidate nodes information I provide that best matches the text to be matched and return its index
Note that the type of action and action object must be exactly matched. for example, '点击消息' and 'clicking to send a message request' can match; '点击report' and 'click share' can not match; '点击消息' and 'click inbox' cannot match.
Some candidate set nodes may appear to have correct action name or description, but the arrival scene is incorrect. you need to choose the correct action node based on arrival scene name and arrival scene description fields.
If you feel that there are some nodes with very similar information, then choose the node with the largest frequency field among these nodes.
## Respond format example
{{
    "index": number // It should be of type Number, which is one of the candidate sets I provided. 
}}
Your response must be in JSON format, that carefully referenced "Respond format example".
If you carefully consider and feel that this action is an input text action, please respond `index: -1` and `input text: xx` in JSON format. Like this:
{{
    "index": -1,
    "input_text: ""
}}
If you have carefully considered and determined that no node in the candidate set matches the action text, please respond `index: 0` in JSON format. Like this:
{{
    "index": 0
}}
---
Next is the data
## complete test case
{full_case}
## The action text to be matched from candidate node
{matching_word}
## candidate nodes
{candidate}
---
"""