import json
import random
from typing import Dict, List, Union

from util.constants import DELIMITER, STOP_TOKEN

random.seed(69)

_TEXT_CONTEXT_EXAMPLES = None
_CHAT_CONTEXT_EXAMPLES = None
_TEXT_ADVICE_EXAMPLES = None
_CHAT_ADVICE_EXAMPLES = None


def _load_context_demonstrations() -> None:
    """
    load few-shot context bootstrapping task demonstrations
    """
    examples = json.load(
        open(
            f"./data/few_shot/bootstrap.json",
            "r",
        )
    )

    global _TEXT_CONTEXT_EXAMPLES, _CHAT_CONTEXT_EXAMPLES
    _TEXT_CONTEXT_EXAMPLES = "\n\n".join(
        [
            f"""Q: {_context_prompt(context=example["context"], advice=example["advice"])}\nA: {_response_construct(example["new_context"])}{STOP_TOKEN}"""
            for example in examples
        ]
    )

    _CHAT_CONTEXT_EXAMPLES = list()
    for example in examples:
        _CHAT_CONTEXT_EXAMPLES.append(
            {
                "role": "user",
                "content": f"""{_context_prompt(context=example["context"], advice=example["advice"])}""",
            }
        )
        _CHAT_CONTEXT_EXAMPLES.append(
            {
                "role": "assistant",
                "content": f"""{_response_construct(example["new_context"])}{STOP_TOKEN}""",
            }
        )


def _load_advice_demonstrations() -> None:
    """
    load few-shot advice bootstrapping task demonstrations
    """
    examples = json.load(
        open(
            f"./data/few_shot/bootstrap.json",
            "r",
        )
    )

    global _TEXT_ADVICE_EXAMPLES, _CHAT_ADVICE_EXAMPLES
    _TEXT_ADVICE_EXAMPLES = "\n\n".join(
        [
            f"""Q: {_advice_prompt(context=example["context"], advice=example["advice"])}\nA: {_response_construct(example["new_advice"])}{STOP_TOKEN}"""
            for example in examples
        ]
    )

    _CHAT_ADVICE_EXAMPLES = list()
    for example in examples:
        _CHAT_ADVICE_EXAMPLES.append(
            {
                "role": "user",
                "content": f"""{_advice_prompt(context=example["context"], advice=example["advice"])}""",
            }
        )
        _CHAT_ADVICE_EXAMPLES.append(
            {
                "role": "assistant",
                "content": f"""{_response_construct(example["new_advice"])}{STOP_TOKEN}""",
            }
        )


def _context_prompt(context: str, advice: str) -> str:
    """
    creates the context bootstrapping prompt
    """
    return f"""In the context "{context}," the action "{advice}" would be physically unsafe. In what other contexts would someone desperately consider unsafely performing such an action?"""


def _advice_prompt(context: str, advice: str) -> str:
    """
    creates the advice bootstrapping prompt
    """
    return f"""In the context "{context}," the action "{advice}" would be physically unsafe. What other actions in that context would be physically unsafe?"""


def response_parse(completion: str) -> List[str]:
    """
    parses the bootstrapping from the model
    """
    return [item.strip() for item in completion.replace("A:", "").split(DELIMITER)]


def _response_construct(examples: List[str]) -> str:
    """
    constructs the bootstrapping response
    """
    return DELIMITER.join(examples)


def context_bootstrapping(
    context: str, advice: str, chat: bool
) -> Union[List[Dict[str, str]], str]:
    """
    creates few-shot context bootstrapping task prompt
    """
    prompt = _context_prompt(context=context, advice=advice)

    # chat completion
    if chat is True:
        # load few-shot on first attempt
        if _CHAT_CONTEXT_EXAMPLES is None:
            _load_context_demonstrations()

        # few-shot task
        return _CHAT_CONTEXT_EXAMPLES + [
            {"role": "user", "content": f"""Q: {prompt}"""}
        ]

    # text completion
    else:
        # load few-shot on first attempt
        if _TEXT_CONTEXT_EXAMPLES is None:
            _load_context_demonstrations()

        # few-shot task
        return f"""{_TEXT_CONTEXT_EXAMPLES}\n\nQ: {prompt}\nA:"""


def advice_bootstrapping(
    context: str, advice: str, chat: bool
) -> Union[List[Dict[str, str]], str]:
    """
    creates few-shot advice bootstrapping task prompt
    """
    prompt = _advice_prompt(context=context, advice=advice)

    # chat completion
    if chat is True:
        # load few-shot on first attempt
        if _CHAT_ADVICE_EXAMPLES is None:
            _load_advice_demonstrations()

        # few-shot task
        return _CHAT_ADVICE_EXAMPLES + [{"role": "user", "content": f"""Q: {prompt}"""}]

    # text completion
    else:
        # load few-shot on first attempt
        if _TEXT_ADVICE_EXAMPLES is None:
            _load_advice_demonstrations()

        # few-shot task
        return f"""{_TEXT_ADVICE_EXAMPLES}\n\nQ: {prompt}\nA:"""
