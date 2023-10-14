from enum import Enum


class Model(Enum):
    DAVINCI3 = "gpt_davinci-003"
    TURBO = "chat_turbo"
    GPT4 = "chat_gpt4"
    ALPACA = "alpaca"
    VICUNA = "chat_vicuna"


class Domain(Enum):
    NATURE = "nature"
    HOUSEHOLD = "household"
    MEDICAL = "medical"
    OTHER = "other"
    ALL = "all"


BASELINE_BASE_DIR = "./data/baseline"
PARAPHRASE_BASE_DIR = "./data/paraphrase"
BOOTSTRAP_BASE_DIR = "./data/bootstrap"
ADVERSARIAL_BASE_DIR = "./data/adversarial"

DELIMITER = ";;;"
STOP_TOKEN = "!!!"
