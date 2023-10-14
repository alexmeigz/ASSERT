# Imports
import json
import os
from typing import List

import numpy as np
import openai
from dotenv import load_dotenv

# Handle environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def check_success(response: any) -> bool:
    """
    checks whether a GPT response was successful
    """
    return type(response) is not dict


def gpt_completion_request(
    prompt: str,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0,
    top_p: float = 1,
    stop_tokens: List[str] = None,
    uncertainty: bool = False,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    **kwargs,
) -> str:
    """
    given a prompt, query gpt-3 as completion task to generate a response and return list of responses.
    max_tokens denotes max response length.
    temperature denotes added randomness in abstractive generation.
    """
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_tokens,
            logprobs=5,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

    except Exception as e:
        print(f"ERROR: {e}")

        return {
            "error": e.__str__(),
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "model": model,
            "stop_tokens": stop_tokens,
            "uncertainty": uncertainty,
        }

    # only return completion in base case
    if uncertainty is False:
        return response["choices"][0]["text"].strip()

    # return completion and uncertainty calculations
    result = list()
    for item in response["choices"]:
        stop_index = max_tokens

        # stop computation at the stop token if it exists
        try:
            stop_index = item["logprobs"]["tokens"].index("<|endoftext|>")
        except:
            pass

        result.append(
            {
                "completion": item["text"].strip(" ."),
                "log_probability": np.sum(
                    item["logprobs"]["token_logprobs"][:stop_index]
                ),
                "first_token_distribution": dict(item["logprobs"]["top_logprobs"][0]),
            }
        )

    return result


def rerun_gpt_queries(filename: str, key: str, transformation) -> None:
    """
    rerun errored gpt completion requests
    """
    # read in old responses
    old_responses = json.load(open(filename, "r"))
    new_responses = list()
    no_errors = True

    for response in old_responses:
        # rerun errored requests
        if check_success(response[key]) is False:
            new_response = gpt_completion_request(**response[key])

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
