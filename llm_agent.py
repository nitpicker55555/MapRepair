import json
import re
import os
from typing import Dict, List, Union, Any

from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

client = OpenAI()


def message_template(role: str, content: str) -> Dict[str, str]:
    """Create a message template dictionary.

    Args:
        role: Message role ('system', 'user', or 'assistant')
        content: Message content

    Returns:
        Dictionary containing role and content
    """
    return {'role': role, 'content': content}


@retry(wait=wait_random_exponential(multiplier=1, max=40),
       stop=stop_after_attempt(3))
def chat_single(messages: List[Dict[str, str]],
                mode: str = "",
                model: str = 'gpt-4o-mini',
                temperature: float = 0,
                verbose: bool = False):
    """Send a single chat request to OpenAI API.

    Args:
        messages: List of messages
        mode: Response mode ('stream', 'json', 'json_few_shot', or empty string for normal mode)
        model: Model to use
        temperature: Temperature parameter, controls response randomness
        verbose: Whether to print detailed information

    Returns:
        Different types of responses based on mode
    """
    if mode == "stream":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            max_tokens=2560
        )
        return response
    elif mode == "json":
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=temperature,
            messages=messages
        )
        return response.choices[0].message.content
    elif mode == 'json_few_shot':
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=2560
        )
        result = response.choices[0].message.content
        if verbose:
            print(result)
        return extract_json_and_similar_words(result)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content


def format_list_string(input_str: str) -> str:
    """格式化包含列表的字符串为有效的JSON。

    Args:
        input_str: 包含列表的字符串

    Returns:
        格式化后的JSON字符串
    """
    match = re.search(r'\{\s*"[^"]+"\s*:\s*\[(.*?)\]\s*\}', input_str)
    if not match:
        return "Invalid input format"

    list_content = match.group(1)
    elements = [e.strip() for e in list_content.split(',')]

    formatted_elements = []
    for elem in elements:
        if not re.match(r'^([\'"])(.*)\1$', elem):
            elem = f'"{elem}"'
        formatted_elements.append(elem)

    return f'{{ "similar_words":[{", ".join(formatted_elements)}]}}'


def extract_json_and_similar_words(text: str) -> Dict[str, Any]:
    """从文本中提取JSON数据。

    Args:
        text: 包含JSON数据的文本

    Returns:
        提取的JSON数据字典
    """
    try:
        json_match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)

        if not json_match:
            raise ValueError("No JSON data found in the text.")

        json_str = json_match.group(1)
        if 'similar_words' in text:
            data = json.loads(format_list_string(json_str))
        else:
            data = json.loads(json_str)

        return data
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return {"error": str(e)}