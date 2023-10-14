import random

from models.eval import compute_accuracy
from models.gpt import check_success
from util.bootstrap_util import (advice_bootstrapping, context_bootstrapping,
                                 response_parse)
from util.constants import BOOTSTRAP_BASE_DIR, STOP_TOKEN, Domain, Model
from util.model_util import completion_request, isChatModel
from util.paraphrase_util import rationale_task
from util.util import base_scenario, load_json, read_base_examples, save_json

random.seed(69)


def bootstrap_context(output_filename: str, model: Model, test: bool = True) -> None:
    """
    bootstraps 5 new contexts for base advice
    """
    # read grounded examples
    examples = read_base_examples(safe=False, domain=Domain.ALL.value)

    # ----- temporary subsampling -----
    if test:
        random.shuffle(examples)
        examples = examples[:10]
    # ---------------------------------

    print(f"RUNNING BOOTSTRAP CONTEXT ON {len(examples)} EXAMPLES")

    for sample in examples:
        # create prompt
        prompt = context_bootstrapping(
            context=sample["prompt"],
            advice=sample["advice"],
            chat=isChatModel(model=model),
        )

        # query language model
        response = completion_request(
            input=prompt,
            model=model,
            max_tokens=256,
            stop_tokens=[STOP_TOKEN],
        )

        # parse response
        if check_success(response) is True:
            sample["new_context"] = response_parse(response)
        else:
            sample["new_context"] = response

    # save response
    save_json(
        response=examples, base_dir=BOOTSTRAP_BASE_DIR, output_filename=output_filename
    )


def bootstrap_advice(
    input_filename: str, output_filename: str, model: Model, test: bool = True
) -> None:
    """
    bootstraps 5 new advice for base context
    """

    # read examples
    examples = load_json(base_dir=BOOTSTRAP_BASE_DIR, input_filename=input_filename)

    # ----- temporary subsampling -----
    if test:
        random.shuffle(examples)
        examples = examples[:10]
    # ---------------------------------

    print(f"RUNNING BOOTSTRAP ADVICE ON {len(examples)} EXAMPLES")

    for sample in examples:
        bootstrap = list()

        for context in sample["new_context"]:
            prompt = advice_bootstrapping(
                context=context,
                advice=sample["advice"],
                chat=isChatModel(model),
            )

            # query language model
            response = completion_request(
                input=prompt,
                model=model,
                max_tokens=256,
                stop_tokens=[STOP_TOKEN],
            )

            # parse response
            if check_success(response) is True:
                for advice in response_parse(response):
                    bootstrap.append(
                        {
                            "prompt": context,
                            "advice": advice,
                        }
                    )

        sample["bootstrap"] = bootstrap

    # save response
    save_json(
        response=examples, base_dir=BOOTSTRAP_BASE_DIR, output_filename=output_filename
    )


def rationalize(
    input_filename: str, output_filename: str, model: Model, test: bool = True
) -> None:
    """
    generates rationales for paraphrases
    """
    # read examples
    examples = load_json(base_dir=BOOTSTRAP_BASE_DIR, input_filename=input_filename)

    # ----- temporary subsampling -----
    if test:
        random.shuffle(examples)
        examples = examples[:10]
    # ---------------------------------
    print(f"RUNNING RATIONALIZE ON {len(examples)} EXAMPLES")

    # generate rationales
    for sample in examples:
        # create prompt
        prompt = rationale_task(
            scenario=base_scenario(prompt=sample["prompt"], advice=sample["advice"]),
            chat=isChatModel(model=model),
        )

        # query language model
        response = completion_request(
            input=prompt,
            model=model,
            max_tokens=128,
        )

        # add rationales
        sample["rationale"] = response

    # save response
    save_json(
        response=examples, base_dir=BOOTSTRAP_BASE_DIR, output_filename=output_filename
    )


def evaluate(input_filename: str, output_filename: str, safe: bool) -> None:
    """
    evaluate paraphrase rationale accuracy
    """
    # read examples
    examples = load_json(base_dir=BOOTSTRAP_BASE_DIR, input_filename=input_filename)

    # evaluate
    save_json(
        response=compute_accuracy(examples=examples, safe=safe),
        base_dir=BOOTSTRAP_BASE_DIR,
        output_filename=output_filename,
    )


if __name__ == "__main__":
    # Example command to run targeted bootstrapping -- context
    bootstrap_context(
        output_filename="bootstrap_context_chatgpt_test", model=Model.TURBO.value
    )

    # Example command to run targeted bootstrapping -- advice
    bootstrap_advice(
        input_filename="bootstrap_context_chatgpt_test",
        output_filename="bootstrap_advice_chatgpt_test",
        model=Model.TURBO.value,
    )

    # Example command to run targeted bootstrapping -- rationale
    rationalize(
        input_filename="bootstrap_samples",
        output_filename="bootstrap_rationale_chatgpt_test",
        model=Model.TURBO.value,
    )

    # Example command to run evaluation
    evaluate(
        input_filename="bootstrap_rationale_chatgpt_test",
        output_filename="eval_unsafe_chatgpt_test",
        safe=False,
    )
