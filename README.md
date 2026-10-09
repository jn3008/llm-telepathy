# LLM Telepathy

A small, reproducible mechanistic interpretability exercise: can a linear monitor predict a frozen language model's next action from its residual stream when the public task and public response do not reveal that action?

This is a controlled experiment, not a claim of original research or human-like mind-reading. The public response is **forced text**: the prompt ends with the exact string `I'm going to choose A.`. We capture activations while processing that prefix, before any action token is generated. The model's generated A/B continuation is the label; the private instruction is only the intended outcome. Refusals and malformed choices stay in the outcome file and are counted separately.

## Experiment

Each public task is paired with an agreeing private instruction and an overriding private instruction. Pair members have identical public task and fixed public response. Public task template families are assigned wholly to train, validation, or test, so paired examples and close paraphrases cannot cross splits. The external monitor sees only the public task and forced response. If actual outcomes are balanced within identical public inputs, chance external accuracy is 50% by construction. That value is a property of the setup, not an experimental finding. The metrics report actual outcome balance and pair balance so this condition is visible. Refusal outputs, other malformed outputs, and valid A/B choices that miss the intended instruction are counted separately.

The internal monitor uses the output of each transformer decoder block at batch index 0 and sequence index -1 of the prompt input. That final position is the final token of the forced response. We run one uncached forward pass to capture these vectors, remove the hooks, and then generate deterministically from the same prompt. Vectors are copied to CPU float32 immediately; only per-example vectors are kept. The run metadata records the layer path, model revision, device, software versions, and token IDs for A/B (with and without a leading space). Generation parsing accepts A or B as the first non-whitespace character and preserves other outputs as malformed.

The probe standardizes features using training data only, picks logistic-regression regularization on validation balanced accuracy, and evaluates each layer once on held-out test data. It reports accuracy, balanced accuracy, pair-cluster bootstrap intervals, class counts, instruction-following rates and failures, majority and shuffled-label controls, a layer plot, and a static test-example viewer. Refusals and malformed outputs are excluded from the binary probe but remain in the counts; instruction-following rates use valid A/B actions as their denominator. If the training split has only one actual class, a constant fallback is reported instead of silently failing or pretending a linear probe was fit.

## Local setup and smoke run

The supplied smoke config produces 36 examples (18 pairs) across the same family-based splits. The full config produces 144 examples (72 pairs). SmolLM2-135M-Instruct is a 135M-parameter, Apache 2.0 decoder-only model with a Transformers chat template. This machine has an Apple M5, 24 GB unified memory, macOS 27, and Python 3.14. Current PyTorch guidance includes Python 3.10–3.14 on macOS and provides MPS acceleration; current scikit-learn releases publish Python 3.14 ARM64 wheels.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m telepathy.dataset --config config.smoke.yaml
python -m telepathy.capture --config config.smoke.yaml
python -m telepathy.probe --config config.smoke.yaml
```

Check the device with `python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())'`; set `model.device` in the config to `auto`, `cpu`, `mps`, or `cuda`.

Outputs are written under `data/` and `artifacts/`. Open `artifacts/smoke_viewer.html` in a browser to browse held-out examples. The smoke split is intentionally small and its confidence intervals will be wide; it checks workflow behavior, not a strong scientific conclusion.

## Larger CUDA run

On a RunPod Linux GPU, create a Python 3.12 environment, install a PyTorch build matching the pod's CUDA driver using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the remaining packages and run:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m telepathy.dataset --config config.yaml
python -m telepathy.capture --config config.yaml
python -m telepathy.probe --config config.yaml
```

Set `device: cuda` in `config.yaml`. This project does not provision GPUs or start paid jobs. For the included 135M model and 144 examples, model weights are roughly 0.3 GB in the repository's BF16 checkpoint; activations are modest (hidden width × layers × examples, stored float32), and inference is sequential. GPU memory needs should be comfortably below 8 GB for this initial config; larger models, sequence lengths, or datasets need fresh estimates. Both configs pin the model to commit `12fd25f77366fa6b3b4b768ec3050bf629380bac`; update the revision deliberately when comparing model versions.

## Interpretation limits

- The model is prompted with a private instruction and then given forced public text. This does not establish spontaneous deception.
- A probe can decode the private instruction itself, task identity, or another correlated feature rather than a distinct internal plan.
- Prediction is correlational. It does not show that a decoded feature causes the generated action.
- This first version does not implement activation patching. If added, patch the donor vector into the live forward pass at the selected block and token position, test both directions, and include unchanged-run and same-action donor controls. Patching near the final block can directly alter next-token preferences.
- Do not drop refusals or malformed generations silently. Inspect them as instruction-following failures and report rates alongside probe scores.
- The family split is a useful first leakage control, but nine template families are too few for broad generalization claims. Expand and pre-register family sets before a larger experiment.

## Reproducibility files

`config.yaml` and `config.smoke.yaml` define the model, seed, split, paths, and probe settings. `data/examples.jsonl` records pair and example IDs, intended choice, template families, and per-example seeds. `artifacts/*outcomes.jsonl` records generated continuations and actual labels; `activations.npz` stores the captured vectors; `run_metadata.json` documents the model and capture point; `metrics.json`, `layer_accuracy.png`, and `viewer.html` are generated by the probe command.

## Compatibility references

- [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [PyTorch install selector and macOS notes](https://pytorch.org/get-started/locally/)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Transformers installation guide](https://huggingface.co/docs/transformers/installation)
