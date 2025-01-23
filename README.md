# ProphetAgent


## About
ProphetAgent can automatically synthesize executable GUI tests from the test steps written in natural language, and then execute the tests on the corresponding apps.
<center><b><font size ='2'>Overview of ProphetAgent</font></b></center>

## Advantages
ProphetAgent has the following advantages as compared with other agents in test case generation and execution:

1. Designed specifically for standardized test cases, offering functionality to synthesize to executable code and execute.
2. Enhances the comprehensive capabilities of LLMs using knowledge graph abilities.
3. Separates code generation from execution, allowing for code review and modification to improve execution success rate.
4. No need to modify the code or system.

## Prerequisite

1. `Python` >= 3.9.6
2. `Java`
3. `Android SDK`

## How to install

Clone this repo and install with `pip`:

```shell
git clone https://github.com/prophetagent/Home
cd ProphetAgent/
pip install -e .
```

## How to use

### Precondition:

 + `.apk` file path of the app you want to analyze.
 + A device or an emulator connected to your host machine via `adb`.
 + Install the neo4j database server.
 + Create a graph database in neo4j and fill the KEYS in config/config.ini

### Start ProphetAgent:
 1. Start GUI Exploration Tool, We have upgraded Droidbot using uiautomator2 to support obtaining various information from dynamic pages like videos
 ```
 droidbot -a <path_to_apk> -o output_dir
 ```
 2. Run SemanticAgent to annotate semantics, clustering, and build graph. Configure the output_dir and database name of neo4j before running
 ```
 python agent_semantic/semantic_agent.py
 ```
 3. Edit the text test case. An example is in `resources/output_example/test_cases/flow_path_example.json`
 4. Run ExecuteAgent to generate execuable code. Configure the path of test cases before running
 ```
 python agent_execute/execute_agent.py
 ```
 The executable code will be generated in the test case directory
 5. Check or modify the code and execute it
 Connect your Android device, install the app, and then run it.
 ```
 python agent_execute/executer.py
 ```

## Acknowledgement

1. [Droidbot](https://github.com/honeynet/droidbot)
2. [Androguard](http://code.google.com/p/androguard/)
3. [uiautomator2](https://github.com/openatx/uiautomator2)

## Useful links

- [Demo video of ProphetAgent](https://www.youtube.com/watch?v=iCsGis__5gg)