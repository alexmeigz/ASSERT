import os
from typing import List

from dotenv import load_dotenv

from models.chatgpt import chat_completion_request, rerun_chat_queries
from models.gpt import gpt_completion_request, rerun_gpt_queries
from models.opensource import FastModel
from util.constants import Model

# Handle environment
load_dotenv()
_CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")

_ALPACA_MODEL = None
_VICUNA_MODEL = None


def _load_alpaca_model() -> None:
    """
    loads alpaca model
    """
    global _ALPACA_MODEL
    _ALPACA_MODEL = FastModel(
        Model.ALPACA.value, cuda_visible_devices=_CUDA_VISIBLE_DEVICES
    )


def _load_vicuna_model() -> None:
    """
    loads vicuna model
    """
    global _VICUNA_MODEL
    _VICUNA_MODEL = FastModel(
        Model.VICUNA.value,
        cuda_visible_devices=_CUDA_VISIBLE_DEVICES,
    )


def isChatModel(model: Model) -> bool:
    """
    returns true if model is chat model
    """
    return "chat" in model


def completion_request(
    input: any,
    model: Model,
    max_tokens: int = 256,
    temperature: float = 0,
    top_p: float = 1,
    stop_tokens: List[str] = None,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    **kwargs,
) -> any:
    if model == Model.GPT4.value:
        return chat_completion_request(
            messages=input,
            model="gpt-4",
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_tokens=stop_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        )

    elif model == Model.TURBO.value:
        return chat_completion_request(
            messages=input,
            model="gpt-3.5-turbo-0301",
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_tokens=stop_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        )

    elif model == Model.DAVINCI3.value:
        return gpt_completion_request(
            prompt=input,
            model="text-davinci-003",
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_tokens=stop_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        )

    elif model == Model.ALPACA.value:
        if _ALPACA_MODEL is None:
            _load_alpaca_model()

        return _ALPACA_MODEL.completion_request(
            message=input,
            model_name=model,
            max_tokens=max_tokens,
            temperature=max(temperature, 0.001),
            top_p=top_p,
        )

    elif model == Model.VICUNA.value:
        if _VICUNA_MODEL is None:
            _load_vicuna_model()

        return _VICUNA_MODEL.completion_request(
            message=input,
            model_name=model,
            max_tokens=max_tokens,
            temperature=max(temperature, 0.001),
            top_p=top_p,
        )

    else:
        raise TypeError(f"unsupported model type: {model}")


def rationale_rerun(base_dir: str, input_filename: str, chat: bool) -> None:
    """
    reruns errored queries
    """
    path = os.path.join(base_dir, input_filename)

    if chat:
        rerun_chat_queries(
            filename=f"{path}.json",
            key="rationale",
            transformation=None,
        )
    else:
        rerun_gpt_queries(
            filename=f"{path}.json",
            key="rationale",
            transformation=None,
        )
