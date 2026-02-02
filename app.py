import os
import time
import torch
import gradio as gr
from diffusers import DiffusionPipeline

# Model: env override (path to snapshot), else HF cache in models_cache, else Hugging Face hub
_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
_MODEL_CACHE = "models_cache"
_SNAPSHOTS_DIR = os.path.join(_MODEL_CACHE, "models--Tongyi-MAI--Z-Image-Turbo", "snapshots")
# Require this file so we use a complete snapshot (not an incomplete one)
_REQUIRED_FILE = os.path.join("text_encoder", "model-00001-of-00003.safetensors")


def _find_complete_snapshot():
    """Pick a snapshot that has full weights (not just refs/index)."""
    if not os.path.isdir(_SNAPSHOTS_DIR):
        return None
    for rev in os.listdir(_SNAPSHOTS_DIR):
        path = os.path.join(_SNAPSHOTS_DIR, rev)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, _REQUIRED_FILE)):
            return path
    return None


if os.environ.get("Z_IMAGE_MODEL_PATH"):
    MODEL_ID = os.environ.get("Z_IMAGE_MODEL_PATH")
    _LOCAL_ONLY = True
elif _find_complete_snapshot() is not None:
    MODEL_ID = _find_complete_snapshot()
    _LOCAL_ONLY = True
else:
    MODEL_ID = _MODEL_ID
    _LOCAL_ONLY = False

# GPU only: require CUDA
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required. Install PyTorch 2.10.0 with CUDA (e.g. pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126).")

# Load the pipeline once at startup
# SDPA uses PyTorch's scaled_dot_product_attention (Flash Attention when available on Space's GPU)
print("Loading Z-Image-Turbo pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    attn_implementation="sdpa",
    local_files_only=_LOCAL_ONLY,
)
pipe.to("cuda")

print("Pipeline loaded!")

def generate_image(prompt, height, width, num_inference_steps, seed, randomize_seed, progress=gr.Progress(track_tqdm=True)):
    """Generate an image from the given prompt."""
    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator("cuda").manual_seed(int(seed))
    t0 = time.perf_counter()
    image = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    elapsed = time.perf_counter() - t0
    print(f"Generated image in {elapsed:.2f}s (seed={seed}, steps={num_inference_steps}, {int(width)}x{int(height)})")
    return image, seed

# Example prompts
examples = [
    ["Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp, bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights."],
    ["A majestic dragon soaring through clouds at sunset, scales shimmering with iridescent colors, detailed fantasy art style"],
    ["Cozy coffee shop interior, warm lighting, rain on windows, plants on shelves, vintage aesthetic, photorealistic"],
    ["Astronaut riding a horse on Mars, cinematic lighting, sci-fi concept art, highly detailed"],
    ["Portrait of a wise old wizard with a long white beard, holding a glowing crystal staff, magical forest background"],
]

# Custom theme with modern aesthetics (Gradio 6)
custom_theme = gr.themes.Soft(
    primary_hue="yellow",
    secondary_hue="amber",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    text_size="lg",
    spacing_size="md",
    radius_size="lg"
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    block_title_text_weight="600",
)

# Build the Gradio interface
with gr.Blocks(fill_height=True) as demo:
    # Header
    gr.Markdown(
        """
        # 🎨 Z-Image-Turbo
        **Ultra-fast AI image generation** • Generate stunning images in just 8 steps
        """,
        elem_classes="header-text"
    )
    
    with gr.Row(equal_height=False):
        # Left column - Input controls
        with gr.Column(scale=1, min_width=320):
            prompt = gr.Textbox(
                label="✨ Your Prompt",
                placeholder="Describe the image you want to create...",
                lines=5,
                max_lines=10,
                autofocus=True,
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    height = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=64,
                        label="Height",
                        info="Image height in pixels"
                    )
                    width = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=64,
                        label="Width",
                        info="Image width in pixels"
                    )
                
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=9,
                    step=1,
                    label="Inference Steps",
                    info="9 steps = 8 DiT forwards (recommended)"
                )
                
                with gr.Row():
                    randomize_seed = gr.Checkbox(
                        label="🎲 Random Seed",
                        value=True,
                    )
                    seed = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0,
                        visible=False,
                    )
                
                def toggle_seed(randomize):
                    return gr.Number(visible=not randomize)
                
                randomize_seed.change(
                    toggle_seed,
                    inputs=[randomize_seed],
                    outputs=[seed]
                )
            
            generate_btn = gr.Button(
                "🚀 Generate Image",
                variant="primary",
                size="lg",
                scale=1
            )
            
            # Example prompts
            gr.Examples(
                examples=examples,
                inputs=[prompt],
                label="💡 Try these prompts",
                examples_per_page=5,
            )
        
        # Right column - Output
        with gr.Column(scale=1, min_width=320):
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                show_label=False,
                height=600,
                buttons=["download", "share"],
            )
            
            used_seed = gr.Number(
                label="🎲 Seed Used",
                interactive=False,
                container=True,
            )
    
    # Footer credits
    gr.Markdown(
        """
        ---
        <div style="text-align: center; opacity: 0.7; font-size: 0.9em; margin-top: 1rem;">
        <strong>Model:</strong> <a href="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo" target="_blank">Tongyi-MAI/Z-Image-Turbo</a> (Apache 2.0 License) • 
        <strong>Demo by:</strong> <a href="https://x.com/realmrfakename" target="_blank">@mrfakename</a> • 
        <strong>Redesign by:</strong> AnyCoder • 
        <strong>Optimizations:</strong> <a href="https://huggingface.co/multimodalart" target="_blank">@multimodalart</a> (FA3 + AoTI)
        </div>
        """,
        elem_classes="footer-text"
    )
    
    # Connect the generate button
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, num_inference_steps, seed, randomize_seed],
        outputs=[output_image, used_seed],
    )
    
    # Also allow generating by pressing Enter in the prompt box
    prompt.submit(
        fn=generate_image,
        inputs=[prompt, height, width, num_inference_steps, seed, randomize_seed],
        outputs=[output_image, used_seed],
    )

if __name__ == "__main__":
    demo.launch(
        theme=custom_theme,
        css="""
        .header-text h1 {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header-text p {
            font-size: 1.1rem !important;
            color: #64748b !important;
            margin-top: 0 !important;
        }
        
        .footer-text {
            padding: 1rem 0;
        }
        
        .footer-text a {
            color: #f59e0b !important;
            text-decoration: none !important;
            font-weight: 500;
        }
        
        .footer-text a:hover {
            text-decoration: underline !important;
        }
        
        /* Mobile optimizations */
        @media (max-width: 768px) {
            .header-text h1 {
                font-size: 1.8rem !important;
            }
            
            .header-text p {
                font-size: 1rem !important;
            }
        }
        
        /* Smooth transitions */
        button, .gr-button {
            transition: all 0.2s ease !important;
        }
        
        button:hover, .gr-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Better spacing */
        .gradio-container {
            max-width: 1400px !important;
            margin: 0 auto !important;
        }
        """,
        footer_links=[
            "api",
            "gradio"
        ],
        mcp_server=True
    )