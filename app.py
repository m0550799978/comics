import os
import json
import re
import uuid
import traceback
import requests
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import InferenceClient
import gradio as gr

# ------------------------------------------------------------------
# Arabic Support Setup
# ------------------------------------------------------------------
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC = True
except ImportError:
    HAS_ARABIC = False
    print("Warning: arabic_reshaper or python-bidi not found.")

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# TEXT: Use "hf-inference" (Hugging Face free serverless API) instead of "auto".
# "auto" routes to paid partners (Novita, Together, etc.) and causes 403 without their keys.
TEXT_PROVIDER = os.environ.get("TEXT_PROVIDER", "together")
TEXT_MODEL_ID = os.environ.get("TEXT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")  # 7B fits free tier, good Arabic

# IMAGE primary: FLUX.2-klein-9B is gated/image-to-image and usually needs a paid provider
IMAGE_MODEL_ID = os.environ.get("IMAGE_MODEL_ID", "black-forest-labs/FLUX.2-klein-9B")
IMAGE_PROVIDER = os.environ.get("IMAGE_PROVIDER", "replicate")  # "auto" (default, may route to paid providers), "hf-inference" (free), "replicate", "fal-ai", "together"
IMAGE_API_KEY = os.environ.get("IMAGE_API_KEY", HF_TOKEN)

# IMAGE fallback: proven text-to-image model on free hf-inference
FALLBACK_PROVIDER = os.environ.get("FALLBACK_PROVIDER", "hf-inference")
FALLBACK_TEXT2IMG_MODEL = os.environ.get("FALLBACK_TEXT2IMG_MODEL", "black-forest-labs/FLUX.1-schnell")
# Alternative if FLUX isn't hosted on your hf-inference region yet:
# FALLBACK_TEXT2IMG_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS



# ------------------------------------------------------------------
# Agents
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Smolagents-style text model wrapper
# ------------------------------------------------------------------
class HFInferenceModel:
    """Callable text backend compatible with agent loops."""
    def __init__(self, model_id: str, token: str = HF_TOKEN, provider: str = TEXT_PROVIDER):
        self.client = InferenceClient(token=token, provider=provider)
        self.model_id = model_id

    def __call__(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024, **kwargs) -> str:
        resp = self.client.chat_completion(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return resp.choices[0].message.content


# ------------------------------------------------------------------
# Agent 1: Writer
# ------------------------------------------------------------------
class WriterAgent:
    def __init__(self, model: HFInferenceModel):
        self.model = model

    def run(self, kid_name: str, kid_age: str, comics_style: str) -> Tuple[List[str], List[str], str]:
        system_msg = (
            "You are an expert Arabic children's comic writer. "
            "You MUST respond with a valid JSON object only."
        )
        user_msg = (
            f"Create a comic for: Name: {kid_name}, Age: {kid_age}, Style: {comics_style}.\n"
            "The JSON must have exactly these keys:\n"
            "1. 'kid_description': A consistent physical description of the child.\n"
            "2. 'arabic_scenes': A list of 10 short Arabic sentences describing the action.\n"
            "3. 'image_prompts': A list of 10 English prompts for an AI artist, each starting with the kid_description.\n"
        )

        # Force the API to return JSON
        response_format = {"type": "json_object"}

        try:
            # We wrap the model call inside the try block to catch API errors too
            raw = self.model(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=2048,
                response_format=response_format,
            )

            # 1. Clean markdown code blocks if present
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            
            # 2. Parse JSON
            data = json.loads(cleaned)
            
            # 3. Extract and pad with defaults if the list is shorter than 10
            arabic_scenes = data.get("arabic_scenes", [])
            image_prompts = data.get("image_prompts", [])
            kid_description = data.get("kid_description", f"a {kid_age} year old child named {kid_name}")

            # Ensure we have exactly 10 items
            final_arabic = (arabic_scenes + ["المشهد القادم..."] * 10)[:10]
            final_prompts = (image_prompts + [f"{kid_description}, comic style"] * 10)[:10]
            
            return final_arabic, final_prompts, kid_description

        except Exception as exc:
            print(f"[WriterAgent] Error encountered: {exc}")
            # Fallback data so the app continues even if AI fails
            fallback_desc = f"a {kid_age} year old child named {kid_name}, {comics_style} style"
            return (
                [f"المشهد {i+1}: مغامرة {kid_name}" for i in range(10)],
                [f"{fallback_desc}, scene {i+1}, highly detailed" for i in range(10)],
                fallback_desc
            )

# ------------------------------------------------------------------
# Agent 2: Artist  (with graceful fallback)
# ------------------------------------------------------------------
class ArtistAgent:
    def __init__(self):
        # Primary: only pass ONE auth arg. token and api_key are aliases in huggingface_hub.
        # For paid providers (replicate, fal-ai), set IMAGE_PROVIDER and IMAGE_API_KEY.
        # For hf-inference (free), just use HF_TOKEN.
        if IMAGE_PROVIDER in ("auto", "hf-inference", "sambanova", "cerebras", "novita", "hyperbolic", "hf-inference"):
            image_auth = HF_TOKEN
        else:
            image_auth = IMAGE_API_KEY

        self.primary_client = InferenceClient(token=image_auth, provider=IMAGE_PROVIDER)
        self.fallback_client = InferenceClient(token=HF_TOKEN, provider=FALLBACK_PROVIDER)

    def run(
        self,
        kid_image_path: str,
        prompts: List[str],
        style: str,
        kid_name: str,
        kid_age: str,
        kid_description: str,
    ) -> Tuple[List[Image.Image], str]:
        images: List[Image.Image] = []
        backend_note = ""

        for i, prompt in enumerate(prompts, start=1):
            full_prompt = (
                f"{style} comic book style illustration. "
                f"Character Profile: {kid_description}. "
                f"Scene Description: {prompt}. "
                f"High quality, consistent character features, vibrant colors, "
                f"bold comic lines, same clothing in every panel."
            )
            print(f"[ArtistAgent] Scene {i}/10: trying image-to-image with {IMAGE_MODEL_ID} "
                  f"(provider={IMAGE_PROVIDER})...")

            result = None

            # ---- ATTEMPT 1: image-to-image (kid photo as reference) ----
            try:
                result = self.primary_client.image_to_image(
                    image=kid_image_path,
                    prompt=full_prompt,
                    model=IMAGE_MODEL_ID,
                    num_inference_steps=4,
                    strength=0.85,
                )
                backend_note = f"Image-to-image via {IMAGE_MODEL_ID} ({IMAGE_PROVIDER})"
            except Exception as exc:
                err_msg = str(exc)
                print(f"[ArtistAgent] image_to_image FAILED: {err_msg[:200]}")
                traceback.print_exc()

                # ---- ATTEMPT 2: text-to-image fallback (free hf-inference) ----
                print(f"[ArtistAgent] Falling back to text-to-image with {FALLBACK_TEXT2IMG_MODEL} "
                      f"(provider={FALLBACK_PROVIDER})...")
                enhanced_prompt = (
                    f"{style} comic book scene. The main character is {kid_description}, age {kid_age}, named {kid_name}. "
                    f"{full_prompt} Character consistency across panels."
                )
                try:
                    result = self.fallback_client.text_to_image(
                        prompt=enhanced_prompt,
                        model=FALLBACK_TEXT2IMG_MODEL,
                        num_inference_steps=4,
                    )
                    backend_note = f"Text-to-image fallback via {FALLBACK_TEXT2IMG_MODEL} ({FALLBACK_PROVIDER})"
                except Exception as exc2:
                    print(f"[ArtistAgent] text_to_image fallback ALSO FAILED: {exc2}")
                    raise RuntimeError(
                        f"Image generation failed completely.\n"
                        f"1) image_to_image failed: {err_msg[:200]}\n"
                        f"2) text_to_image fallback failed: {exc2}\n\n"
                        f"TIPS:\n"
                        f"- If using FLUX.2-klein-9B, you likely need a paid provider (replicate/fal-ai) + IMAGE_API_KEY.\n"
                        f"- Make sure you accepted the model license at https://hf.co/{IMAGE_MODEL_ID}\n"
                        f"- Or switch IMAGE_MODEL_ID/FALLBACK_TEXT2IMG_MODEL to 'stabilityai/stable-diffusion-xl-base-1.0'"
                    ) from exc2

            if result is None:
                raise RuntimeError("No image was generated.")
            if isinstance(result, bytes):
                result = Image.open(BytesIO(result))
            images.append(result)

        return images, backend_note


# ------------------------------------------------------------------
# Agent 3: Editor (PDF Compiler)
# ------------------------------------------------------------------
class EditorAgent:
    def __init__(self):
        # Force it to use the file you uploaded to Hugging Face
        self.font_path = "Amiri-Regular.ttf" 
        
    def _reshape_only(self, text: str) -> str:
        """Connects the letters without reversing the order. Used for measuring width."""
        if HAS_ARABIC and text.strip():
            return arabic_reshaper.reshape(text)
        return text

    def _shape_for_display(self, text: str) -> str:
        """Connects letters AND reverses for RTL. Used for actual drawing."""
        if HAS_ARABIC and text.strip():
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        return text

    def _get_font(self, size: int):
        """Helper to ensure we load the font correctly."""
        try:
            if os.path.exists(self.font_path):
                return ImageFont.truetype(self.font_path, size)
        except Exception as e:
            print(f"Font error: {e}")
        return ImageFont.load_default()

    def _create_title_page(self, kid_name: str, phone: str, style: str, backend_note: str) -> Image.Image:
        W, H = 1080, 1528
        page = Image.new("RGB", (W, H), (20, 24, 40))
        draw = ImageDraw.Draw(page)

        big = self._get_font(90)
        med = self._get_font(50)
        small = self._get_font(32)

        title = self._shape_for_display(f"مغامرات {kid_name}")
        draw.text((W // 2, H // 3), title, fill=(255, 215, 0), font=big, anchor="mm")

        sub = self._shape_for_display(f"نمط القصص المصورة: {style}")
        draw.text((W // 2, H // 2), sub, fill=(220, 220, 220), font=med, anchor="mm")

        contact = self._shape_for_display(f"للتواصل: {phone}")
        draw.text((W // 2, int(H * 0.58)), contact, fill=(180, 180, 180), font=small, anchor="mm")
        
        # Note: Backend note is usually English, so no shaping needed
        draw.text((W // 2, int(H * 0.85)), f"Generated with: {backend_note}", fill=(120, 120, 120), font=small, anchor="mm")

        return page

    def _create_page(self, image: Image.Image, scene_text: str, page_num: int, kid_name: str) -> Image.Image:
        W, H = 1080, 1528
        page = Image.new("RGB", (W, H), (255, 250, 230))
        draw = ImageDraw.Draw(page)

        header_font = self._get_font(44)
        text_font = self._get_font(38)

        # Header
        header = self._shape_for_display(f"قصة {kid_name} - الصفحة {page_num}")
        draw.text((W // 2, 50), header, fill=(139, 0, 0), font=header_font, anchor="mm")

        # Image Panel
        panel_w, panel_h = 1000, 950
        scene_img = image.convert("RGB")
        scene_img.thumbnail((panel_w, panel_h), LANCZOS)
        page.paste(scene_img, ((W - scene_img.width) // 2, 140))

        # --- FOCUSED FIX: ARABIC WORD WRAPPING ---
        words = scene_text.split()
        lines: List[str] = []
        current_line_words = []
        max_line_width = W - 150 # Margin

        for w in words:
            test_line = " ".join(current_line_words + [w])
            # MEASURE using Reshaped text (connected) but NOT Reversed yet
            reshaped_test = self._reshape_only(test_line)
            bbox = draw.textbbox((0, 0), reshaped_test, font=text_font)
            
            if bbox[2] - bbox[0] <= max_line_width:
                current_line_words.append(w)
            else:
                if current_line_words:
                    lines.append(" ".join(current_line_words))
                current_line_words = [w]
        if current_line_words:
            lines.append(" ".join(current_line_words))

        # --- FOCUSED FIX: DRAWING ---
        y_cursor = 1130
        for ln in lines:
            # ONLY apply BiDi flip (get_display) at the final drawing step
            shaped_line = self._shape_for_display(ln)
            draw.text((W // 2, y_cursor), shaped_line, fill=(20, 20, 20), font=text_font, anchor="mm")
            y_cursor += 55

        return page

    def run(
        self,
        images: List[Image.Image],
        arabic_scenes: List[str],
        kid_name: str,
        phone: str,
        style: str,
        backend_note: str,
        output_path: str = "comic_book.pdf",
    ) -> str:
        pages: List[Image.Image] = []
        pages.append(self._create_title_page(kid_name, phone, style, backend_note).convert("RGB"))

        for idx, (img, txt) in enumerate(zip(images, arabic_scenes), start=1):
            pages.append(self._create_page(img, txt, idx, kid_name).convert("RGB"))

        pages[0].save(output_path, save_all=True, append_images=pages[1:], resolution=150.0)
        return output_path

# ------------------------------------------------------------------
# Manager Agent (Orchestrator)
# ------------------------------------------------------------------
@dataclass
class ComicState:
    kid_name: str = ""
    kid_age: str = ""
    comics_style: str = ""
    phone_number: str = ""
    kid_image_path: str = ""
    arabic_scenes: List[str] = field(default_factory=list)
    image_prompts: List[str] = field(default_factory=list)
    kid_description: str = ""
    generated_images: List[Image.Image] = field(default_factory=list)
    backend_note: str = ""
    pdf_path: str = ""


class ManagerAgent:
    def __init__(self, hf_token: str = HF_TOKEN):
        text_backend = HFInferenceModel(TEXT_MODEL_ID, token=hf_token, provider=TEXT_PROVIDER)
        self.writer = WriterAgent(text_backend)
        self.artist = ArtistAgent()
        self.editor = EditorAgent()

    def run(
        self,
        kid_name: str,
        kid_age: str,
        comics_style: str,
        phone_number: str,
        kid_image_path: str,
    ) -> Tuple[List[Image.Image], str, str, str]:
        state = ComicState(
            kid_name=kid_name, kid_age=kid_age, comics_style=comics_style,
            phone_number=phone_number, kid_image_path=kid_image_path,
        )

        # Stage 1: Writer
        print("[Manager] === Stage 1: Writer Agent ===")
        state.arabic_scenes, state.image_prompts, state.kid_description = self.writer.run(
            state.kid_name, state.kid_age, state.comics_style
        )
        print(f"[Manager] Story drafted. Kid desc: {state.kid_description}")

        # Stage 2: Artist
        print("[Manager] === Stage 2: Artist Agent ===")
        state.generated_images, state.backend_note = self.artist.run(
            kid_image_path=state.kid_image_path,
            prompts=state.image_prompts,
            style=state.comics_style,
            kid_name=state.kid_name,
            kid_age=state.kid_age,
            kid_description=state.kid_description,
        )
        print(f"[Manager] Images generated via: {state.backend_note}")

        # Stage 3: Editor
        print("[Manager] === Stage 3: Editor Agent ===")
        pdf_file = f"comic_{state.kid_name.strip().replace(' ', '_') or 'kid'}.pdf"
        state.pdf_path = self.editor.run(
            state.generated_images,
            state.arabic_scenes,
            state.kid_name,
            state.phone_number,
            state.comics_style,
            state.backend_note,
            output_path=pdf_file,
        )
        print("[Manager] Comic book compiled.")

        return state.generated_images, state.pdf_path, "\n\n".join(state.arabic_scenes), state.backend_note



# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------
app = FastAPI()
# Lazy initialization to prevent startup crash
manager = None

def get_manager():
    global manager
    if manager is None:
        manager = ManagerAgent()
    return manager

@app.post("/webhook")
async def webhook(
    background_tasks: BackgroundTasks,
    kid_name: str = Form(...),
    kid_age: str = Form(...),
    comics_style: str = Form(..., alias="comics-style"),
    phone_number: str = Form(..., alias="phone-number"),
    kid_image: UploadFile = File(...)
):
    m = get_manager()
    job_id = uuid.uuid4().hex
    temp_path = f"tmp_{job_id}.png"
    with open(temp_path, "wb") as f:
        f.write(await kid_image.read())
    
    background_tasks.add_task(m.run, kid_name, kid_age, comics_style, phone_number, temp_path)
    return {"status": "Accepted", "job_id": job_id}

# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------
def build_ui():
    def process_ui(name, age, style, phone, img):
        m = get_manager()
        path = "ui_tmp.png"
        img.save(path)
        return m.run(name, age, style, phone, path)

    with gr.Blocks() as demo:
        gr.Markdown("# 🦸‍♀️ Comic Factory")
        with gr.Row():
            with gr.Column():
                n = gr.Textbox(label="Name")
                a = gr.Dropdown(["3-5", "5-7", "7-10"], label="Age")
                s = gr.Dropdown(["Manga", "Marvel", "Disney"], label="Style")
                p = gr.Textbox(label="Phone")
                i = gr.Image(type="pil", label="Kid Image")
                btn = gr.Button("Generate")
            with gr.Column():
                gal = gr.Gallery()
                pdf = gr.File()
                txt = gr.Textbox()
        btn.click(process_ui, [n, a, s, p, i], [gal, pdf, txt])
    return demo

app = gr.mount_gradio_app(app, build_ui(), path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
