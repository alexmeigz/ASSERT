import os
from typing import List

import torch
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.inference import load_model

from util.constants import Model

# Handle environment
load_dotenv()
VICUNA_MODEL_PATH = os.getenv("VICUNA_MODEL_PATH")
ALPACA_MODEL_PATH = os.getenv("ALPACA_MODEL_PATH")


class FastModel:
    def __init__(
        self,
        model_name,
        model_device="cuda",
        cuda_visible_devices=[0, 1, 2],
        max_gpu_memory=None,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    ):
        model_path = None
        if model_name == Model.VICUNA.value:
            model_path = VICUNA_MODEL_PATH
        elif model_name == Model.ALPACA.value:
            model_path = ALPACA_MODEL_PATH
        else:
            raise ValueError(
                f"Model name {model_name} not recognized. Choose between 'vicuna' or 'alpaca'."
            )

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in cuda_visible_devices]
        )
        num_gpus = len(cuda_visible_devices)

        self.model, self.tokenizer = load_model(
            model_path,
            model_device,
            num_gpus,
            max_gpu_memory,
            load_8bit,
            cpu_offloading,
            debug,
        )

    def completion_request(
        self,
        message: any,
        model_name: str,
        max_tokens: int = 256,
        temperature: float = 0.001,
        top_p: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
    ):
        if model_name == Model.VICUNA.value:
            conv = get_conversation_template(VICUNA_MODEL_PATH)
            for i, m in enumerate(message[:-1]):
                # print(i, type(i), m, type(m))
                conv.append_message(conv.roles[i % 2], m["content"])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

        elif model_name == Model.ALPACA.value:
            conv = get_conversation_template(ALPACA_MODEL_PATH)
            conv.append_message(conv.roles[0], message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

        else:
            raise ValueError(
                f"Model name {model_name} not recognized. Choose between 'vicuna' or 'alpaca'."
            )

        input_ids = self.tokenizer([prompt]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=top_p,
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        return outputs
