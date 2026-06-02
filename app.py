"""Single-turn compare lab for control-vector injection.

Type a prompt, set a strength slider per vector, and watch the controlled
response stream next to the unsteered baseline. This is the interactive version
of the `for i in range(-10, 10, 2)` sweep in explore.py / explore_risk.py.
"""

import threading

import gradio as gr
from transformers import TextIteratorStreamer

from vectors import chatml, load_model_and_vectors

print("Loading model and training/caching vectors (first run is slow)...")
model, tokenizer, specs = load_model_and_vectors()
print("Ready.")

def gen_settings(temperature, top_p, repetition_penalty):
    """Build generate() kwargs. temperature 0 = greedy, matching the scripts."""
    settings = dict(
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=float(repetition_penalty),
    )
    if temperature and temperature > 0:
        settings.update(do_sample=True, temperature=float(temperature), top_p=float(top_p))
    else:
        settings.update(do_sample=False)
    return settings

# One GPU: serialize generation across requests/panels.
_gpu_lock = threading.Lock()

EXAMPLE_PROMPT = (
    "I have $10,000 in savings. Should I invest it all in a risky new startup, "
    "or keep it in the bank? Tell me what to do."
)


def _stream(prompt: str, control_vector, max_new_tokens, gen_kwargs):
    """Yield the growing decoded completion for one prompt + optional vector."""
    input_ids = tokenizer(chatml(prompt), return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    with _gpu_lock:
        model.reset()
        if control_vector is not None:
            model.set_control(control_vector, 1.0)
        thread = threading.Thread(
            target=model.generate,
            kwargs={
                **input_ids,
                **gen_kwargs,
                "max_new_tokens": int(max_new_tokens),
                "streamer": streamer,
            },
        )
        thread.start()
        text = ""
        for token in streamer:
            text += token
            yield text
        thread.join()
        model.reset()


def run(
    prompt,
    show_baseline,
    max_new_tokens,
    temperature,
    top_p,
    repetition_penalty,
    *strengths,
):
    """Stream baseline (optional) and controlled responses side by side."""
    prompt = (prompt or "").strip()
    if not prompt:
        yield "", "*Enter a prompt above.*"
        return

    gen_kwargs = gen_settings(temperature, top_p, repetition_penalty)

    # Blend the active vectors: combined = sum(vector * slider_strength)
    combined = None
    for spec, strength in zip(specs, strengths):
        if strength:
            term = spec.vector * float(strength)
            combined = term if combined is None else combined + term

    label = " ".join(f"{s.name}={v:+g}" for s, v in zip(specs, strengths)) or "all zero"

    baseline_text = ""
    if show_baseline:
        for baseline_text in _stream(prompt, None, max_new_tokens, gen_kwargs):
            yield baseline_text, "*generating…*"
    else:
        baseline_text = "*(baseline disabled)*"

    controlled = ""
    for controlled in _stream(prompt, combined, max_new_tokens, gen_kwargs):
        yield baseline_text, controlled
    yield baseline_text, controlled + f"\n\n— `{label}`"


with gr.Blocks(title="Control Vector Lab") as demo:
    gr.Markdown(
        "# Control Vector Lab\n"
        "Set a strength per vector, then **Generate** to compare the steered "
        "response against the unsteered baseline. Positive = "
        + ", ".join(f"**{s.positive_persona}**" for s in specs)
        + "; negative = "
        + ", ".join(f"**{s.negative_persona}**" for s in specs)
        + "."
    )

    prompt = gr.Textbox(
        label="Prompt", value=EXAMPLE_PROMPT, lines=2, placeholder="Ask something…"
    )

    sliders = []
    with gr.Row():
        for spec in specs:
            sliders.append(
                gr.Slider(
                    minimum=spec.suggested_range[0],
                    maximum=spec.suggested_range[1],
                    value=0.0,
                    step=0.5,
                    label=f"{spec.name}  ({spec.negative_persona} ⟷ {spec.positive_persona})",
                )
            )

    with gr.Accordion("Advanced settings", open=False):
        max_new_tokens = gr.Slider(
            minimum=32,
            maximum=1024,
            value=256,
            step=32,
            label="Max output tokens",
        )
        temperature = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=0.0,
            step=0.05,
            label="Temperature (0 = greedy / deterministic)",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling; only used when temperature > 0)",
        )
        repetition_penalty = gr.Slider(
            minimum=1.0,
            maximum=2.0,
            value=1.1,
            step=0.05,
            label="Repetition penalty",
        )

    with gr.Row():
        show_baseline = gr.Checkbox(value=True, label="Show baseline")
        generate = gr.Button("Generate", variant="primary")

    with gr.Row():
        baseline_out = gr.Markdown(label="Baseline")
        controlled_out = gr.Markdown(label="Controlled")

    generate.click(
        run,
        inputs=[
            prompt,
            show_baseline,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
            *sliders,
        ],
        outputs=[baseline_out, controlled_out],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
