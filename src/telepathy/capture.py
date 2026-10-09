from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from telepathy.common import ensureParent, loadConfig, readJsonl, writeJson


action_pattern = re.compile(r"^\s*([AB])(?:\b|[.!?])", re.IGNORECASE)
refusal_pattern = re.compile(r"^\s*(?:i\s+(?:cannot|can't|won't|will not)|sorry\b|as an ai\b|i must refuse)\b", re.IGNORECASE)


def chooseDevice(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def getDecoderLayers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        current = model
        try:
            for segment in path.split("."):
                current = getattr(current, segment)
            return current, path
        except AttributeError:
            continue
    raise RuntimeError("Could not locate decoder blocks; add the model architecture's layer path.")


def parseAction(generated_text: str) -> str:
    if refusal_pattern.match(generated_text):
        return "refusal"
    match = action_pattern.match(generated_text)
    return match.group(1).upper() if match else "malformed"


def makePrompt(tokenizer, row: dict) -> str:
    user_text = (
        f"Public task: {row['public_task']}\n"
        f"Private instruction: {row['private_instruction']}\n"
        "After your public acknowledgment, make the requested final choice."
    )
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + row["forced_public_response"]


def captureRow(model, tokenizer, layers, row: dict, device: torch.device, max_new_tokens: int):
    prompt = makePrompt(tokenizer, row)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    captured = {}
    handles = []

    def saveLastToken(layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Decoder block output is [batch, sequence, hidden]. Position -1
            # is the final token of the forced public response in this input.
            captured[layer_index] = hidden[0, -1, :].detach().to("cpu", dtype=torch.float32).numpy()
        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(saveLastToken(index)))
    with torch.inference_mode():
        model(**inputs, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    actual_choice = parseAction(generated_text)
    result = dict(row)
    result.update({
        "actual_choice": actual_choice,
        "generated_text": generated_text,
        "instruction_followed": actual_choice == row["intended_choice"],
        "outcome_status": "valid_choice" if actual_choice in ("A", "B") else actual_choice,
        "prompt_token_count": int(inputs["input_ids"].shape[1]),
    })
    return result, captured


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen-model inference and capture public-prefix activations.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N example cap for a quick smoke run.")
    args = parser.parse_args()
    config = loadConfig(args.config)
    rows = readJsonl(config["paths"]["examples"])
    if args.limit:
        rows = rows[:args.limit]
    device = chooseDevice(config["model"].get("device", "auto"))
    model_name = config["model"]["name"]
    revision = config["model"].get("revision", "main")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers, layer_path = getDecoderLayers(model)

    # Preserve tokenizer behavior explicitly: an action may be a single token,
    # multiple tokens, or include leading-space variants depending on context.
    choice_tokenization = {
        choice: {
            "bare_token_ids": tokenizer.encode(choice, add_special_tokens=False),
            "space_prefixed_token_ids": tokenizer.encode(" " + choice, add_special_tokens=False),
        }
        for choice in ("A", "B")
    }

    captured_rows = []
    activation_rows = [[] for _ in layers]
    for index, row in enumerate(rows, start=1):
        result, captured = captureRow(
            model, tokenizer, layers, row, device, int(config["model"]["max_new_tokens"])
        )
        captured_rows.append(result)
        for layer_index in range(len(layers)):
            activation_rows[layer_index].append(captured[layer_index])
        if index % 10 == 0 or index == len(rows):
            print(f"Processed {index}/{len(rows)} examples")

    activation_path = config["paths"]["activations"]
    ensureParent(activation_path)
    arrays = {f"layer_{index}": np.stack(values).astype(np.float32) for index, values in enumerate(activation_rows)}
    np.savez_compressed(activation_path, **arrays)
    output_rows_path = str(Path(activation_path).with_suffix(".outcomes.jsonl"))
    with open(output_rows_path, "w", encoding="utf-8") as output_file:
        for row in captured_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    writeJson(config["paths"]["run_metadata"], {
        "model": model_name,
        "revision": revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "decoder_layer_path": layer_path,
        "layer_count": len(layers),
        "activation_site": "each decoder block output, batch 0, sequence position -1; the last token of the forced public response before any action tokens are generated",
        "choice_tokenization": choice_tokenization,
        "decoding": {"do_sample": False, "max_new_tokens": int(config["model"]["max_new_tokens"])},
        "action_parse_rule": "first non-whitespace output character must be A or B; refusal prefixes and all other outputs are labeled separately",
        "dataset_seed": config["dataset"]["seed"],
        "activation_file": activation_path,
        "outcomes_file": output_rows_path,
        "examples_captured": len(captured_rows),
    })
    print(f"Saved activations to {activation_path}")
    print(f"Saved outcomes to {output_rows_path}")


if __name__ == "__main__":
    main()
