"""
Needle-in-a-Haystack test for TurboQuant.

Hides a specific fact in a long document and checks if the model can retrieve it.
This is the paper's flagship benchmark (0.997 recall at 4x compression).
"""

import sys
sys.path.insert(0, "/home/azureuser/turboquant")

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from turboquant.cache import TurboQuantCache

NEEDLE = "The secret code for the treasure chest is BLUE-DRAGON-42."

HAYSTACK_UNIT = (
    "The history of artificial intelligence began in antiquity, with myths and stories of "
    "artificial beings endowed with intelligence by master craftsmen. Classical philosophers "
    "attempted to describe the process of human thinking as the mechanical manipulation of "
    "symbols. This work culminated in the invention of the programmable digital computer in "
    "the 1940s. Alan Turing proposed that machines could simulate any conceivable act of "
    "mathematical reasoning. The field of AI research was founded at a workshop at Dartmouth "
    "College in 1956. Early AI programs solved algebra problems, proved theorems, and learned "
    "to speak English. By the mid-1960s, research was heavily funded by the Department of "
    "Defense. In the 1970s, AI faced criticism and funding cuts known as the AI winter. "
    "Expert systems were developed in the 1980s, and neural networks regained popularity. "
    "Deep learning breakthroughs in the 2010s led to dramatic advances in computer vision "
    "and natural language processing. Today, AI powers search engines, recommendation systems, "
    "autonomous vehicles, and language models that can generate human-like text. "
)

QUESTION = "What is the secret code for the treasure chest?"


def build_prompt(context_tokens, tokenizer, needle_position=0.5):
    """Build a prompt with a needle hidden in a haystack at the given position."""
    # Build haystack
    haystack_tokens = tokenizer.encode(HAYSTACK_UNIT)
    needle_tokens = tokenizer.encode(NEEDLE)
    target_hay_tokens = context_tokens - len(needle_tokens) - 50  # leave room for question

    n_repeats = target_hay_tokens // len(haystack_tokens) + 1
    full_haystack = HAYSTACK_UNIT * n_repeats

    # Truncate to target length
    hay_encoded = tokenizer.encode(full_haystack)[:target_hay_tokens]

    # Insert needle at position
    insert_idx = int(len(hay_encoded) * needle_position)
    combined = hay_encoded[:insert_idx] + needle_tokens + hay_encoded[insert_idx:]
    combined_text = tokenizer.decode(combined)

    prompt = f"{combined_text}\n\nBased on the text above, answer this question: {QUESTION}"
    return prompt


def test_needle(model, tokenizer, context_length, needle_position=0.5, use_turboquant=False, skip_layers=None):
    """Run one needle test and check if the model retrieves the answer."""
    prompt = build_prompt(context_length, tokenizer, needle_position)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_length).to(model.device)
    actual_len = inputs.input_ids.shape[1]

    if use_turboquant:
        cache = TurboQuantCache(model.config, nbits=4, residual_length=128,
                                device="cuda", skip_layers=skip_layers or set())
    else:
        cache = None

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=50, do_sample=False,
            past_key_values=cache,
        )
    answer = tokenizer.decode(output[0][actual_len:], skip_special_tokens=True)

    # Check if the needle info is in the answer
    found = "BLUE-DRAGON-42" in answer or "BLUE" in answer and "DRAGON" in answer and "42" in answer
    return {
        "context_length": actual_len,
        "needle_position": needle_position,
        "found": found,
        "answer": answer[:200],
    }


def main():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        ),
    )
    print(f"Loaded: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    skip = TurboQuantCache.calibrate_skip_layers(model, tokenizer)
    print(f"Skip layers: {skip}")

    context_lengths = [1024, 2048, 4096, 8192, 16384]
    positions = [0.25, 0.5, 0.75]

    print(f"\n{'Context':>8} {'Position':>8} | {'Default':>10} {'TurboQuant':>12} | {'Match':>6}")
    print("-" * 60)

    total_default = 0
    total_tq = 0
    total_tests = 0

    for ctx in context_lengths:
        for pos in positions:
            # Default
            r_default = test_needle(model, tokenizer, ctx, pos, use_turboquant=False)
            gc.collect(); torch.cuda.empty_cache()

            # TurboQuant
            r_tq = test_needle(model, tokenizer, ctx, pos, use_turboquant=True, skip_layers=skip)
            gc.collect(); torch.cuda.empty_cache()

            match = r_default["found"] == r_tq["found"]
            total_default += r_default["found"]
            total_tq += r_tq["found"]
            total_tests += 1

            d_str = "FOUND" if r_default["found"] else "MISS"
            t_str = "FOUND" if r_tq["found"] else "MISS"
            m_str = "=" if match else "DIFF"

            print(f"{r_default['context_length']:>8} {pos:>8.2f} | {d_str:>10} {t_str:>12} | {m_str:>6}")

            if not r_tq["found"]:
                print(f"         TQ answer: {r_tq['answer'][:80]}")

    print(f"\nResults: Default {total_default}/{total_tests}, TurboQuant {total_tq}/{total_tests}")
    print(f"Default recall:    {100*total_default/total_tests:.1f}%")
    print(f"TurboQuant recall: {100*total_tq/total_tests:.1f}%")


if __name__ == "__main__":
    main()
