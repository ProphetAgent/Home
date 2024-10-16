import json
import re
import time
import openai
import configparser
import base64
from colorama import Fore
from typing import List


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ask_gpt4o(system_prompt, user_prompt, images: List[str], need_json=True) -> dict:
    config = configparser.ConfigParser()
    config.read('../config/config.ini')

    # 获取GPT API密钥和endpoint
    ak = config.get('gpt4', 'gpt_key')
    endpoint = config.get('gpt4', 'endpoint')
    content = [
        {
            "type": "text",
            "text": user_prompt
        }
    ]
    for img in images:
        base64_img = encode_image(img)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })
    api_version = "2024-03-01-preview"
    ak = ak
    model_name = "gpt-4o-2024-05-13"
    max_tokens = 4096
    client = openai.AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        api_key=ak,
    )

    max_attempts = 5
    attempt = 0
    have_answer = False
    while attempt < max_attempts:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=max_tokens
            )
            have_answer = True
        except Exception as e:
            print(Fore.RED + "gpt4o Exception: " + str(e) + Fore.RESET)
            if 'qpm limit' in str(e):
                time.sleep(20)
            else:
                time.sleep(5)
            attempt += 1
            if attempt == max_attempts:
                return str(e) + "gpt ans failed, parse failed"
        if have_answer:
            response = json.loads(completion.model_dump_json())
            # print(Fore.YELLOW + response['choices'][0]['message']['content'] + Fore.RESET)
            if not need_json:
                return response['choices'][0]['message']['content']
            else:
                pattern = r'\{[^}]*\}'
                try:
                    matches = re.findall(pattern, response['choices'][0]['message']['content'])
                    # Assuming we want the first match (in this case, there is only one)
                    if matches:
                        extracted_info = matches[0]
                        return json.loads(extracted_info)
                    else:
                        attempt += 1
                except Exception as e:
                    print(Fore.RED + "re match Exception: " + str(e) + Fore.RESET)
        else:
            attempt += 1
    return "None"
