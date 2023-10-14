import random

from models.eval import compute_accuracy
from models.gpt import check_success
from util.adversarial_util import adversarial_task, internal_probing
from util.bootstrap_util import response_parse
from util.constants import ADVERSARIAL_BASE_DIR, Domain, Model
from util.model_util import completion_request, isChatModel
from util.util import load_json, read_base_examples, save_json

random.seed(69)


def adversarial_probing(output_filename: str, model: Model, test: bool = True) -> None:
    """
    generates 5 hypothetical benefits for unsafe scenario
    """
    # read grounded examples
    examples = read_base_examples(safe=False, domain=Domain.ALL.value)

    # ----- temporary subsampling -----
    if test:
        random.shuffle(examples)
        examples = examples[:10]
    # ---------------------------------

    print(f"RUNNING INTERNAL PROBING ON {len(examples)} EXAMPLES")

    for sample in examples:
        # create prompt
        prompt = internal_probing(
            context=sample["prompt"],
            advice=sample["advice"],
            chat=isChatModel(model),
        )

        # query language model
        response = completion_request(
            input=prompt,
            model=model,
            max_tokens=128,
        )

        # parse response
        if check_success(response) is True:
            sample["benefits"] = response_parse(response)
        else:
            sample["benefits"] = response

    # save response
    save_json(
        response=examples,
        base_dir=ADVERSARIAL_BASE_DIR,
        output_filename=output_filename,
    )


def format_examples(input_filename: str, output_filename: str) -> None:
    """
    reformat paraphrase examples
    """
    # read examples
    examples = load_json(base_dir=ADVERSARIAL_BASE_DIR, input_filename=input_filename)

    # format examples
    new_samples = list()
    for sample in examples:
        if len(sample["benefits"]) == 1:
            print(sample["benefits"])
        for benefit in sample["benefits"]:
            new_samples.append(
                {
                    "prompt": sample["prompt"],
                    "advice": sample["advice"],
                    "benefit": benefit,
                    "domain": sample["domain"],
                }
            )

    # save examples
    save_json(
        response=new_samples,
        base_dir=ADVERSARIAL_BASE_DIR,
        output_filename=output_filename,
    )


def adversarial_suite(
    input_filename: str,
    output_filename: str,
    model: Model,
    demo: bool,
    test: bool = True,
) -> None:
    """
    generates rationales for paraphrases
    """
    # read examples
    examples = load_json(base_dir=ADVERSARIAL_BASE_DIR, input_filename=input_filename)

    # ----- temporary subsampling -----
    if test:
        random.shuffle(examples)
        examples = examples[:10]
    # ---------------------------------
    print(f"RUNNING ADVERSARIAL SUITE ON {len(examples)} EXAMPLES")

    # generate rationales
    for sample in examples:
        # create prompt
        prompt = adversarial_task(
            context=sample["prompt"],
            advice=sample["advice"],
            knowledge=sample["benefit"],
            chat=isChatModel(model=model),
            demo=demo,
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
        response=examples,
        base_dir=ADVERSARIAL_BASE_DIR,
        output_filename=output_filename,
    )


def evaluate(input_filename: str, output_filename: str, safe: bool) -> None:
    """
    evaluate paraphrase rationale accuracy
    """
    # read examples
    examples = load_json(base_dir=ADVERSARIAL_BASE_DIR, input_filename=input_filename)

    # evaluate
    save_json(
        response=compute_accuracy(examples=examples, safe=safe),
        base_dir=ADVERSARIAL_BASE_DIR,
        output_filename=output_filename,
    )


if __name__ == "__main__":
    # Example command to generate adversarial knowledge
    adversarial_probing(
        output_filename="raw_benefits_unsafe_gpt3_test", model=Model.DAVINCI3.value
    )

    # Example command to format adversarial knowledge injection examples
    format_examples(
        input_filename="raw_benefits_unsafe_gpt3_test",
        output_filename="benefits_unsafe_gpt3_test",
    )

    # Example command to run adversarial knowledge injection
    adversarial_suite(
        input_filename="benefits_unsafe_gpt3_test",
        output_filename="rationale_unsafe_gpt3_test",
        model=Model.DAVINCI3.value,
        demo=True,  # set to True to add adversarial demonstrations
    )

    # Example command to run evaluation
    evaluate(
        input_filename="rationale_unsafe_gpt3_test",
        output_filename="rationale_unsafe_gpt3_test",
        safe=False,
    )
