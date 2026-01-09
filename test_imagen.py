"""
Test Google Imagen 4 via Replicate
"""
import replicate
import os

# Set API token
os.environ["REPLICATE_API_TOKEN"] = "r8_GDM8eRWquSvM7P99tZ31QdACNe1COSN42SVeS"

# Test prompt
input_data = {
    "prompt": "A cinematic shot of a beautiful sunset over the ocean, golden light reflecting on the water, waves gently crashing on the shore, photorealistic, 8k, ultra detailed",
    "aspect_ratio": "16:9",
    "safety_filter_level": "block_medium_and_above"
}

print("🎨 Generating image with Google Imagen 4...")
print(f"Prompt: {input_data['prompt'][:100]}...")

try:
    output = replicate.run(
        "google/imagen-4",
        input=input_data
    )
    
    # Get the URL
    print(f"\n✅ Image generated!")
    print(f"URL: {output.url}")
    
    # Save to file
    output_path = "test_imagen_output.png"
    with open(output_path, "wb") as file:
        file.write(output.read())
    
    print(f"📁 Saved to: {output_path}")
    print(f"\n🎉 Success! Open {output_path} to view the result.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
