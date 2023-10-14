import random

from models.eval import compute_accuracy
from util.constants import BASELINE_BASE_DIR, Domain, Model
from util.model_util import completion_request, isChatModel
from util.paraphrase_util import rationale_task
from util.util import base_scenario, load_json, read_base_examples, save_json

random.seed(69)


def rationalize(
    safe: bool, output_filename: str, model: Model, test: bool = True
) -> None:
    """
    generates rationales for paraphrases
    """
    # read grounded examples
    examples = read_base_examples(safe=safe, domain=Domain.ALL.value)

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
        response=examples, base_dir=BASELINE_BASE_DIR, output_filename=output_filename
    )


def evaluate(input_filename: str, output_filename: str, safe: bool) -> None:
    """
    evaluate paraphrase rationale accuracy
    """
    # read examples
    examples = load_json(base_dir=BASELINE_BASE_DIR, input_filename=input_filename)

    # evaluate
    save_json(
        response=compute_accuracy(examples=examples, safe=safe),
        base_dir=BASELINE_BASE_DIR,
        output_filename=output_filename,
    )


if __name__ == "__main__":
    # Example command to run baseline rationale generation
    rationalize(
        safe=False,
        output_filename="rationales_unsafe_gpt3_test",
        model=Model.DAVINCI3.value,
        test=True,
    )

    # Example command to run baseline evaluation
    evaluate(
        input_filename="rationales_unsafe_gpt3_test",
        output_filename="eval_unsafe_gpt3_test",
        safe=False,
    )
