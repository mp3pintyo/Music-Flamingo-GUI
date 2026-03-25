from __future__ import annotations

from typing import Any

import gradio as gr

from music_flamingo_gui.formatting import summarize_user_message
from music_flamingo_gui.inference import DEFAULT_MODEL_ID, GenerationOptions, ModelOptions, get_service


def _quantization_value(label: str) -> str:
    return "4bit" if label == "4-bit NF4" else "full"


def load_model(model_id: str, quantization_label: str) -> str:
    try:
        options = ModelOptions(
            model_id=model_id.strip() or DEFAULT_MODEL_ID,
            quantization=_quantization_value(quantization_label),
        )
        return get_service().load_model(options)
    except Exception as exc:
        return f"Hiba a modell betöltése közben: {exc}"


def unload_model() -> str:
    return get_service().unload_model()


def clear_conversation() -> tuple[list[dict[str, str]], list[dict[str, Any]], str, str, str, None]:
    return [], [], "", "", "Beszélgetés törölve.", None


def submit_message(
    chatbot_history: list[dict[str, str]],
    conversation_history: list[dict[str, Any]],
    prompt: str,
    audio_path: str | None,
    model_id: str,
    quantization_label: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    # Azonnal jelezzük a felhasználónak, hogy a kéréskérdés beérkezett.
    user_display = summarize_user_message(prompt or "", audio_path)
    pending_chat = list(chatbot_history)
    pending_chat.append({"role": "user", "content": user_display})
    pending_chat.append({"role": "assistant", "content": "\u23f3 Feldolgozás folyamatban..."})

    # Figyelmeztető üzenet, ha az audio 30 másodpercnél hosszabb.
    trunc_warning = ""
    if audio_path:
        try:
            import librosa
            _dur = librosa.get_duration(path=audio_path)
            if _dur > 30.0:
                trunc_warning = f" ⚠️ Az audio {_dur:.1f}s hosszú — a modell csak az első 30s-t elemzi (a processzor jelenlegi limitje)."
        except Exception:
            pass

    yield pending_chat, conversation_history, "", "", "Az audió feldolgozása és a válasz generálása elkezdődött..." + trunc_warning, audio_path

    try:
        options = ModelOptions(
            model_id=model_id.strip() or DEFAULT_MODEL_ID,
            quantization=_quantization_value(quantization_label),
        )
        generation = GenerationOptions(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        service = get_service()

        final_conv = conversation_history
        final_reasoning = ""
        final_answer = ""
        final_status = "Válasz elkészült."

        for update in service.generate_stream(prompt, audio_path, conversation_history, options, generation):
            if update.get("done"):
                final_conv = update["conversation"]
                final_reasoning = update["reasoning"]
                final_answer = update["answer"]
                final_status = update.get("status_note", "Válasz elkészült.")
            else:
                content = update.get("answer") or update.get("full_text") or "..."
                pending_chat[-1]["content"] = content
                yield list(pending_chat), conversation_history, update.get("reasoning", ""), content, "Generálás folyamatban...", audio_path

        final_chat = list(chatbot_history)
        final_chat.append({"role": "user", "content": user_display})
        final_chat.append({"role": "assistant", "content": final_answer})
        yield final_chat, final_conv, final_reasoning, final_answer, final_status + trunc_warning, None

    except Exception as exc:
        pending_chat[-1]["content"] = f"Hiba: {exc}"
        yield list(pending_chat), conversation_history, "", "", f"Hiba: {str(exc)[:500]}", audio_path


def build_ui() -> tuple[gr.Blocks, gr.themes.Theme, str]:
    app_css = """
    .app-shell {max-width: 1440px; margin: 0 auto;}
    .hero {background: linear-gradient(135deg, #07111f 0%, #0f2740 45%, #173d57 100%); color: #f6f7eb; border-radius: 24px; padding: 24px;}
    .hero h1, .hero p {margin: 0; color: white !important;}
    .note {border-left: 4px solid #dca54c; padding-left: 12px;}
    """

    app_theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="amber",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Music Flamingo Local GUI") as demo:
        chatbot_state = gr.State([])
        conversation_state = gr.State([])

        gr.HTML(
            """
            <div class="app-shell">
              <div class="hero">
                <h1>Music Flamingo Local GUI</h1>
                <p>Helyi webes felület az nvidia/music-flamingo-think-2601-hf modellhez, audió és szöveges kérdésekkel.</p>
              </div>
            </div>
            """
        )
        gr.Markdown(
            "A modell nem kereskedelmi kutatási licenc alatt érhető el. Első indításnál a súlyok letöltése és a modell betöltése percekig is tarthat."
        )

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label="Beszélgetés", height=620)
                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="Utasítás",
                        placeholder="Példa: Elemezd a dal műfaját, tempóját, hangnemét és a hangszerelését.",
                        lines=4,
                    )
                audio_input = gr.Audio(label="Audió feltöltés", type="filepath", sources=["upload"])
                with gr.Row():
                    submit_button = gr.Button("Küldés", variant="primary")
                    clear_button = gr.Button("Beszélgetés törlése")
            with gr.Column(scale=5):
                model_id = gr.Textbox(label="Modell", value=DEFAULT_MODEL_ID)
                quantization = gr.Radio(
                    label="Memória mód",
                    choices=["4-bit NF4", "BF16 / FP16"],
                    value="4-bit NF4",
                )
                with gr.Row():
                    load_button = gr.Button("Modell betöltése")
                    unload_button = gr.Button("Modell kiürítése")
                max_new_tokens = gr.Slider(label="Max új token", minimum=64, maximum=2048, step=64, value=1024)
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.5, step=0.05, value=0.6)
                top_p = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, step=0.05, value=0.9)
                status_box = gr.Markdown(value="Várakozás a modell betöltésére.")
                gr.Markdown("A 4 bites opció nem GGUF, hanem Transformers-alapú NF4 kvantálás. A loader alapból 12 GiB GPU plafonnal és CPU offloaddal indul, mert a teljes modell 16 GB VRAM-ba sem mindig fér be teljesen.")
                gr.Markdown("### Gondolatmenet")
                reasoning_box = gr.Markdown()
                gr.Markdown("### Végső válasz")
                answer_box = gr.Markdown()

        submit_inputs = [
            chatbot_state,
            conversation_state,
            prompt_input,
            audio_input,
            model_id,
            quantization,
            max_new_tokens,
            temperature,
            top_p,
        ]
        submit_outputs = [chatbot, conversation_state, reasoning_box, answer_box, status_box, audio_input]

        submit_button.click(submit_message, inputs=submit_inputs, outputs=submit_outputs).then(
            lambda current_chat: current_chat,
            inputs=[chatbot],
            outputs=[chatbot_state],
        ).then(lambda: "", outputs=[prompt_input])

        prompt_input.submit(submit_message, inputs=submit_inputs, outputs=submit_outputs).then(
            lambda current_chat: current_chat,
            inputs=[chatbot],
            outputs=[chatbot_state],
        ).then(lambda: "", outputs=[prompt_input])

        clear_button.click(
            clear_conversation,
            outputs=[chatbot, conversation_state, reasoning_box, answer_box, status_box, audio_input],
        ).then(lambda: [], outputs=[chatbot_state])

        load_button.click(load_model, inputs=[model_id, quantization], outputs=[status_box])
        unload_button.click(unload_model, outputs=[status_box])

    return demo, app_theme, app_css


if __name__ == "__main__":
    demo, app_theme, app_css = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=app_theme, css=app_css)

