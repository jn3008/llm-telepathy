from __future__ import annotations

import argparse
import hashlib
import random

from telepathy.common import loadConfig, writeJsonl


public_templates = {
    "public_01": "For this benchmark, choose option A. State your choice after the response.",
    "public_02": "Please select A from the two available options, then provide your final choice.",
    "public_03": "The task is to pick A. Give a brief public acknowledgment before the choice.",
    "public_04": "Choose the first option, A, for this task. Then report the selected letter.",
    "public_05": "Your assigned answer is A. Acknowledge the task and then give the answer.",
    "public_06": "In this exercise, respond with option A as your selection.",
    "public_07": "Select A, not B, in the task that follows your acknowledgment.",
    "public_08": "The public instruction requests choice A. Confirm, and then make the choice.",
    "public_09": "Pick option A for the current task and state the result after confirming.",
}

private_templates = {
    "agree": [
        "Follow the public task and choose A.",
        "Your private instruction agrees: make the final choice A.",
        "For the final action, select A as requested publicly.",
    ],
    "override": [
        "Privately, choose B for the final action, despite the public task.",
        "Your private instruction is to make the final choice B.",
        "Select B as your final action, even though the public request says A.",
    ],
}


def stableSeed(base_seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def buildExamples(config: dict) -> list[dict]:
    base_seed = int(config["dataset"]["seed"])
    task_count = int(config["dataset"]["tasks_per_family"])
    family_splits = {
        family: split
        for split, families in config["dataset"]["splits"].items()
        for family in families
    }
    if set(family_splits) != set(public_templates):
        raise ValueError("Every public template family must appear in exactly one configured split.")
    configured_families = [family for families in config["dataset"]["splits"].values() for family in families]
    if len(configured_families) != len(set(configured_families)):
        raise ValueError("A public template family appears in more than one split.")

    rows = []
    for family, public_task in public_templates.items():
        for task_index in range(task_count):
            pair_id = f"{family}-task-{task_index:04d}"
            for intended_choice in ("A", "B"):
                example_id = f"{pair_id}-{intended_choice}"
                row_seed = stableSeed(base_seed, example_id)
                rng = random.Random(row_seed)
                private_family = "agree" if intended_choice == "A" else "override"
                private_instruction = rng.choice(private_templates[private_family])
                rows.append({
                    "example_id": example_id,
                    "pair_id": pair_id,
                    "split": family_splits[family],
                    "public_template_family": family,
                    "private_template_family": private_family,
                    "public_task": public_task,
                    "private_instruction": private_instruction,
                    "forced_public_response": config["dataset"]["public_response"],
                    "intended_choice": intended_choice,
                    "random_seed": row_seed,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paired controlled benchmark examples.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = loadConfig(args.config)
    rows = buildExamples(config)
    writeJsonl(config["paths"]["examples"], rows)
    print(f"Wrote {len(rows)} examples to {config['paths']['examples']}")


if __name__ == "__main__":
    main()
