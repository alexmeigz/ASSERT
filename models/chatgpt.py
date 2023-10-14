# Imports
import json
import os
from typing import Dict, List

import openai
from dotenv import load_dotenv

from models.gpt import check_success

# Handle environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat_completion_request(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 256,
    temperature: float = 0,
    top_p: float = 1,
    stop_tokens: List[str] = None,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    **kwargs,
) -> str:
    """
    given a prompt, query chat model as completion task to generate a response and return list of responses.
    max_tokens denotes max response length.
    temperature denotes added randomness in abstractive generation.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

    except Exception as e:
        print(f"ERROR: {e}")

        # return function parameters as dictionary
        return {
            "error": e.__str__(),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "model": model,
            "stop_tokens": stop_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

    return response["choices"][0]["message"]["content"].strip("A: ")


def rerun_chat_queries(filename: str, key: str, transformation) -> None:
    """
    rerun errored chat completion requests
    """
    # read in old responses
    old_responses = json.load(open(filename, "r"))
    new_responses = list()
    no_errors = True

    for response in old_responses:
        # rerun errored requests
        if check_success(response[key]) is False:
            new_response = chat_completion_request(**response[key])

            # if new response is successful, apply transformation
            if check_success(new_response) is True and transformation is not None:
                new_response = transformation(new_response)

            new_responses.append(
                {
                    **response,
                    key: new_response,
                }
            )
            no_errors = False
        # keep responses that didn't error
        else:
            new_responses.append(response)

    # verbose output
    if no_errors:
        print("No errors found!")

    # write error fixes to same file
    json.dump(new_responses, open(filename, "w"), indent=2)


if __name__ == "__main__":
    print(
        chat_completion_request(
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {
                    "role": "assistant",
                    "content": "The Los Angeles Dodgers won the World Series in 2020.",
                },
                {"role": "user", "content": "Where was it played?"},
            ],
        )
    )
