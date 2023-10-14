import random

from models.eval import compute_accuracy
from models.gpt import check_success
from util.bootstrap_util import response_parse
from util.constants import PARAPHRASE_BASE_DIR, STOP_TOKEN, Domain, Model
from util.model_util import completion_request, isChatModel
from util.paraphrase_util import paraphrase_task, rationale_task
from util.util import load_json, read_base_examples, save_json

random.seed(69)


def identify_paraphrase(
    output_filename: str, model: Model, safe: bool, test: bool = True
) -> None:
    """
    generate five semantically aligned prompts from base prompt
    """
    # read grounded examples
    examples = read_base_examples(safe=safe, domain=Domain.ALL.value)

    # ----- temporary subsampling -----
    if test:
        random.shuffle(examples)
        examples = examples[:10]
    # ---------------------------------

    print(f"RUNNING SEMANTIC ON {len(examples)} EXAMPLES")

    for sample in examples:
        # create prompt
        prompt = paraphrase_task(
            context=sample["prompt"],
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
            sample["paraphrase"] = response_parse(response)
        else:
            sample["paraphrase"] = response

    # save response
    save_json(
        response=examples, base_dir=PARAPHRASE_BASE_DIR, output_filename=output_filename
    )


def format_examples(input_filename: str, output_filename: str) -> None:
    """
    reformat paraphrase examples into a list of dicts with a single paraphrase in each
    """
    # read examples
    examples = load_json(base_dir=PARAPHRASE_BASE_DIR, input_filename=input_filename)

    # format examples
    new_samples = list()
    for sample in examples:
        if len(sample["paraphrase"]) == 1:
            print(sample["paraphrase"])

        for paraphrase in sample["paraphrase"]:
            if "?" not in paraphrase:
                print(paraphrase)
            else:
                new_samples.append(
                    {
                        "paraphrase": paraphrase.replace("!!!", ""),
                        "domain": sample["domain"],
                    }
                )
    print(len(new_samples))

    # save examples
    save_json(
        response=new_samples,
        base_dir=PARAPHRASE_BASE_DIR,
        output_filename=output_filename,
    )


def rationalize(
    input_filename: str, output_filename: str, model: Model, test: bool = True
) -> None:
    """
    generates rationales for paraphrases
    """
    # read examples
    examples = load_json(base_dir=PARAPHRASE_BASE_DIR, input_filename=input_filename)

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
            scenario=sample["paraphrase"],
            chat=isChatModel(model),
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
        response=examples, base_dir=PARAPHRASE_BASE_DIR, output_filename=output_filename
    )


def evaluate(input_filename: str, output_filename: str, safe: bool) -> None:
    """
    evaluate paraphrase rationale accuracy
    """
    # read examples
    examples = load_json(base_dir=PARAPHRASE_BASE_DIR, input_filename=input_filename)

    # evaluate
    save_json(
        response=compute_accuracy(examples=examples, safe=safe),
        base_dir=PARAPHRASE_BASE_DIR,
        output_filename=output_filename,
    )


if __name__ == "__main__":
    # Example command to run semantically aligned augmentation
    identify_paraphrase(
        output_filename="raw_paraphrases_unsafe_gpt3_test",
        model=Model.DAVINCI3.value,
        safe=False,
    )

    # Example command to format semantically aligned augmentation examples
    format_examples(
        input_filename="raw_paraphrases_unsafe_gpt3_test",
        output_filename="paraphrases_unsafe_gpt3_test",
    )

    # Example command to run rationale generation for semantically aligned augmentation
    rationalize(
        input_filename="paraphrases_unsafe_gpt3_test",
        output_filename="rationales_unsafe_gpt3_test",
        model=Model.DAVINCI3.value,
    )

    # Example command to run evaluation
    evaluate(
        input_filename="rationales_unsafe_gpt3_test",
        output_filename="eval_unsafe_gpt3_test",
        safe=False,
    )
