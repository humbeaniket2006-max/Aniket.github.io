

from optimum.intel import OVStableDiffusionPipeline
import torch

# 1. SETUP: Load the "Lite" Model
# We use the FP16 (half-precision) version. 
# This reduces RAM usage from ~6GB to ~2-3GB[cite: 34].
model_id = "OpenVINO/stable-diffusion-v1-5-fp16-ov"

print("Loading Neural Engine (this may take 1-2 minutes)...")

# 2. OPTIMIZATION: Disable Compilation
# compile=False prevents a long "warmup" time on the first run[cite: 34].
gen = OVStableDiffusionPipeline.from_pretrained(model_id, compile=False)

# 3. CONSTRAINTS: Lock Resolution and Batch Size
# We lock the input size to 512x512. 
# Batch size is set to 1 to prevent 'Killed: 9' (Out of Memory) errors[cite: 31, 34].
gen.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)

# 4. HARDWARE TARGET: CPU
# We explicitly target "cpu" because the Intel Iris GPU often causes crashes 
# due to shared memory limits[cite: 34, 65].
gen.to("cpu") 

print("Starting generation... (Expect 45-90 seconds per image)")

# 5. GENERATION LOOP
# We run inference with reduced steps (20) to avoid thermal throttling[cite: 35].
images = gen(
    prompt="human iris, brown, macro photography, 8k, detailed",
    negative_prompt="blurry, drawing, painting, illustration",
    num_inference_steps=20, 
    height=512, 
    width=512
).images

# 6. SAVE OUTPUT
# Saves the result to the experiments folder we created[cite: 35].
for idx, img in enumerate(images):
    save_path = f"./experiments/baseline_lite_{idx}.png"
    img.save(save_path)
    print(f"Success! Image saved to: {save_path}")