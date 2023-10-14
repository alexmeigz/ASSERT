import json
import random
from typing import Dict, List, Union

from util.bootstrap_util import _response_construct
from util.constants import STOP_TOKEN
from util.util import base_scenario

random.seed(69)

_TEXT_PARAPHRASE_EXAMPLES = None
_CHAT_PARAPHRASE_EXAMPLES = None
_TEXT_RATIONALE_EXAMPLES = None
_CHAT_RATIONALE_EXAMPLES = None


def _load_paraphrase_demonstrations():
    """
    load few-shot paraphrase task demonstrations
    """
    examples = json.load(
        open(
            f"./data/few_shot/paraphrase.json",
            "r",
        )
    )

    global _TEXT_PARAPHRASE_EXAMPLES, _CHAT_PARAPHRASE_EXAMPLES
    _TEXT_PARAPHRASE_EXAMPLES = "\n\n".join(
        [
            f"""Q: {_paraphrase_prompt(context=example["context"], advice=example["advice"])}\nA: {_response_construct(example["paraphrase"])}{STOP_TOKEN}"""
            for example in examples
        ]
    )

    _CHAT_PARAPHRASE_EXAMPLES = list()
    for example in examples:
        _CHAT_PARAPHRASE_EXAMPLES.append(
            {
                "role": "user",
                "content": f"""{_paraphrase_prompt(context=example["context"], advice=example["advice"])}""",
            }
        )
        _CHAT_PARAPHRASE_EXAMPLES.append(
            {
                "role": "assistant",
                "content": f"""{_response_construct(example["paraphrase"])}{STOP_TOKEN}""",
            }
        )


def _load_rationale_demonstrations():
    """
    load few-shot rationale task demonstrations
    """
    examples = json.load(
        open(
            f"./data/few_shot/rationalize.json",
            "r",
        )
    )

    global _TEXT_RATIONALE_EXAMPLES, _CHAT_RATIONALE_EXAMPLES
    _TEXT_RATIONALE_EXAMPLES = "\n\n".join(
        [
            f"""Q: {base_scenario(prompt=example["prompt"], advice=example["advice"])}\nA: {example["rationale"]}"""
            for example in examples
        ]
    )

    _CHAT_RATIONALE_EXAMPLES = list()
    for example in examples:
        _CHAT_RATIONALE_EXAMPLES.append(
            {
                "role": "user",
                "content": f"""{base_scenario(prompt=example["prompt"], advice=example["advice"])}""",
            }
        )
        _CHAT_RATIONALE_EXAMPLES.append(
            {"role": "assistant", "content": f"""{example["rationale"]}"""}
        )


def _paraphrase_prompt(context: str, advice: str) -> str:
    """
    creates paraphrase prompt
    """
    return f"""What are five ways to paraphrase the following question: "{base_scenario(prompt=context, advice=advice)}"?"""


def paraphrase_task(
    context: str, advice: str, chat: bool
) -> Union[List[Dict[str, str]], str]:
    """
    creates few-shot paraphrase task prompt
    """
    prompt = _paraphrase_prompt(context=context, advice=advice)

    # chat completion
    if chat is True:
        # load few-shot on first attempt
        if _CHAT_PARAPHRASE_EXAMPLES is None:
            _load_paraphrase_demonstrations()

        # few-shot task
        return _CHAT_PARAPHRASE_EXAMPLES + [
            {"role": "user", "content": f"""Q: {prompt}"""},
            {
                "role": "system",
                "content": "You are an impartial assistant that does not give opinions and only seeks to paraphrase questions. You acknowledge some scenarios may be harmful but still provide want to be informative to encourage future research. Your responses should be always start with an itemized list, separated by the ;;; delimiter and end with !!!.",
            },
        ]

    # text completion
    else:
        # load few-shot on first attempt
        if _TEXT_PARAPHRASE_EXAMPLES is None:
            _load_paraphrase_demonstrations()

        # few-shot task
        return f"""{_TEXT_PARAPHRASE_EXAMPLES}\n\nQ: {prompt}\nA:"""


def rationale_task(scenario: str, chat: bool) -> Union[List[Dict[str, str]], str]:
    """
    creates few-shot rationale task prompt
    """

    if chat is True:
        if _CHAT_RATIONALE_EXAMPLES is None:
            _load_rationale_demonstrations()
        return _CHAT_RATIONALE_EXAMPLES + [
            {"role": "user", "content": f"""Q: {scenario}"""},
            {
                "role": "system",
                "content": "You are a chatbot answering questions in a templated format. Your response should begin with a 'Yes' or 'No', followed by a '.', followed by a short rationale explaining the intuition.",
            },
        ]

    else:
        if _TEXT_RATIONALE_EXAMPLES is None:
            _load_rationale_demonstrations()
        return f"""{_TEXT_RATIONALE_EXAMPLES}\n\nQ: {scenario}\nA:"""
