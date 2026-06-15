import os
import sys
import subprocess
import time

# 3. Add to path
sys.path.append(os.path.abspath("LuxTTS"))

import numpy as np
import gradio as gr
import torch
from zipvoice.luxvoice import LuxTTS

# Init Model
device = "cuda" if torch.cuda.is_available() else "cpu"
lux_tts = LuxTTS("YatharthS/LuxTTS", device=device, threads=2)

def infer(
    text,
    audio_prompt,
    rms,
    ref_duration,
    t_shift,
    num_steps,
    speed,
    return_smooth,
):
    if audio_prompt is None or not text:
        return None, "Please provide text and reference audio."

    start_time = time.time()

    # Encode reference (WITH duration)
    encoded_prompt = lux_tts.encode_prompt(
        audio_prompt,
        duration=ref_duration,
        rms=rms,
    )

    # Generate speech
    final_wav = lux_tts.generate_speech(
        text,
        encoded_prompt,
        num_steps=int(num_steps),
        t_shift=t_shift,
        speed=speed,
        return_smooth=return_smooth,
    )

    duration = round(time.time() - start_time, 2)

    final_wav = final_wav.cpu().squeeze(0).numpy()
    final_wav = (np.clip(final_wav, -1.0, 1.0) * 32767).astype(np.int16)

    stats_msg = f"✨ Generation complete in **{duration}s**."
    return (48000, final_wav), stats_msg

# =======================
# Gradio UI
# =======================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ LuxTTS Voice Cloning")

    gr.Markdown(
        """
        > **Note:** This demo runs on a **2-core CPU**, so expect slower inference.  
        > **Tip:** If words get cut off, lower **Speed** or increase **Ref Duration**.
        """
    )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Text to Synthesize",
                value="Hey, what's up? I'm feeling really great!",
            )
            input_audio = gr.Audio(
                label="Reference Audio (.wav)",
                type="filepath",
            )

            with gr.Row():
                rms_val = gr.Number(
                    value=0.01,
                    label="RMS (Loudness)",
                )
                ref_duration_val = gr.Number(
                    value=5,
                    label="Reference Duration (sec)",
                    info="Lower = faster. Set ~1000 if you hear artifacts.",
                )
                t_shift_val = gr.Number(
                    value=0.9,
                    label="T-Shift",
                )

            with gr.Row():
                steps_val = gr.Slider(
                    1,
                    10,
                    value=4,
                    step=1,
                    label="Num Steps",
                )
                speed_val = gr.Slider(
                    0.5,
                    2.0,
                    value=0.8,
                    step=0.1,
                    label="Speed (Lower = Longer / Clearer)",
                )
                smooth_val = gr.Checkbox(
                    label="Return Smooth",
                    value=False,
                )

            btn = gr.Button("Generate Speech", variant="primary")

        with gr.Column():
            audio_out = gr.Audio(label="Result")
            status_text = gr.Markdown("Ready to generate...")

    btn.click(
        fn=infer,
        inputs=[
            input_text,
            input_audio,
            rms_val,
            ref_duration_val,
            t_shift_val,
            steps_val,
            speed_val,
            smooth_val,
        ],
        outputs=[audio_out, status_text],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)

