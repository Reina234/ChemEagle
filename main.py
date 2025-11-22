import sys
import torch
import json
from chemietoolkit import ChemIEToolkit,utils
import cv2
from openai import AzureOpenAI
import numpy as np
from PIL import Image
import json
import os
import sys
from rxnim import RxnIM
import json
import base64

from get_molecular_agent import process_reaction_image_with_multiple_products_and_text_correctR
from get_reaction_agent import get_reaction_withatoms_correctR
from get_R_group_sub_agent import process_reaction_image_with_table_R_group, process_reaction_image_with_product_variant_R_group,get_full_reaction_template,get_multi_molecular_full
from get_observer import action_observer_agent, plan_observer_agent
from get_text_agent import text_extraction_agent


model = ChemIEToolkit(device=torch.device('cpu')) 
ckpt_path = "./rxn.ckpt"
model1 = RxnIM(ckpt_path, device=torch.device('cpu'))
device = torch.device('cpu')

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

def ChemEagle(
    image_path: str,
    use_plan_observer: bool = False,
    use_action_observer: bool = False,
) -> dict:

    client = AzureOpenAI(
        api_key=API_KEY,
        api_version='2024-06-01',
        azure_endpoint=AZURE_ENDPOINT
    )

    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    tools = [
        {
        'type': 'function',
        'function': {
            'name': 'process_reaction_image_with_product_variant_R_group',
            'description': 'get the reaction data of the reaction diagram and get SMILES strings of every detailed reaction in reaction diagram and the set of product variants, and the original molecular list.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
            {
        'type': 'function',
        'function': {
            'name': 'process_reaction_image_with_table_R_group',
            'description': 'get the reaction data of the reaction diagram and get SMILES strings of every detailed reaction in reaction diagram and the R-group table',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
            {
        'type': 'function',
        'function': {
            'name': 'get_full_reaction_template',
            'description': 'After you carefully check the image, if this is a reaction image that contains only a text-based table and does not involve any R-group replacement, or this is a reaction image does not contain any tables or sets of product variants, then just call this simplified tool.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
            {
        'type': 'function',
        'function': {
            'name': 'get_multi_molecular_full',
            'description': 'After you carefully check the image, if this is a single molecule image or a multiple molecules image, then need to call this molecular recognition tool.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
    ]

    # 提供给 GPT 的消息内容
    with open('./prompt/prompt_final_simple_version.txt', 'r',encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()
    with open('./prompt/prompt_plan.txt', 'r',encoding='utf-8') as prompt_file:
        prompt_plan = prompt_file.read()

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
            ]
        }
    ]

    # 调用 GPT 接口
    response = client.chat.completions.create(
    model = 'gpt-4o',
    temperature = 0,
    response_format={ 'type': 'json_object' },
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt_plan
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{base64_image}'
                    }
                }
            ]},
    ],
    tools = tools)
    
# Step 1: 工具映射表
    TOOL_MAP = {
        'process_reaction_image_with_product_variant_R_group': process_reaction_image_with_product_variant_R_group,
        'process_reaction_image_with_table_R_group': process_reaction_image_with_table_R_group,
        'get_full_reaction_template': get_full_reaction_template,
        'get_multi_molecular_full': get_multi_molecular_full
    }

    # Step 2: 处理多个工具调用
    raw_tool_calls = response.choices[0].message.tool_calls or []
    serialized_calls = []
    for idx, tool_call in enumerate(raw_tool_calls):
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        serialized_calls.append({
            "id": getattr(tool_call, "id", f"tool_call_{idx}"),
            "name": tool_call.function.name,
            "arguments": args,
        })

    if use_plan_observer:
        reviewed_plan = plan_observer_agent(image_path, serialized_calls)
        if not isinstance(reviewed_plan, list) or not reviewed_plan:
            plan_to_execute = serialized_calls
        else:
            plan_to_execute = []
            for idx, item in enumerate(reviewed_plan):
                name = item.get("name") or item.get("tool_name")
                if not name:
                    continue
                args = item.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call_id = item.get("id") or f"observer_call_{idx}"
                plan_to_execute.append({
                    "id": call_id,
                    "name": name,
                    "arguments": args,
                })
            if not plan_to_execute:
                plan_to_execute = serialized_calls
    else:
        plan_to_execute = serialized_calls

    print(f"plan_to_execute:{plan_to_execute}")
    execution_logs = []
    results = []

    for idx, plan_item in enumerate(plan_to_execute):
        tool_name = plan_item["name"]
        tool_call_id = plan_item["id"] or f"observer_call_{idx}"
        tool_args = plan_item.get("arguments", {}) or {}
        if "image_path" not in tool_args:
            tool_args["image_path"] = image_path

        if tool_name in TOOL_MAP:
            tool_func = TOOL_MAP[tool_name]
            tool_result = tool_func(**tool_args)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")

        execution_logs.append({
            "id": tool_call_id,
            "name": tool_name,
            "arguments": tool_args,
            "result": tool_result,
        })

        results.append({
            'role': 'tool',
            'content': json.dumps({
                'image_path': image_path,
                tool_name: tool_result,
            }),
            'tool_call_id': tool_call_id,
        })

    if use_action_observer and action_observer_agent(image_path, execution_logs):
        return {
            "redo": True,
            "plan": plan_to_execute,
            "execution_logs": execution_logs,
        }

    observer_plan_message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": item["id"],
                "type": "function",
                "function": {
                    "name": item["name"],
                    "arguments": json.dumps(item.get("arguments", {})),
                },
            }
            for item in plan_to_execute
        ],
    }


# Prepare the chat completion payload
    completion_payload = {
        'model': 'gpt-4o',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{base64_image}'
                        }
                    }
                ]
            },
            observer_plan_message,
            *results
            ],
    }

# Generate new response
    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        response_format={ 'type': 'json_object' },
        temperature=0
    )


    
    # 获取 GPT 生成的结果
    gpt_output = json.loads(response.choices[0].message.content)
    print(gpt_output)
    return gpt_output

if __name__ == "__main__":
    model = ChemEagle()
