"""
Simple script to create a minimal ONNX model for testing SAMO
This avoids the Windows path length issues with full TensorFlow installation
"""

# Try to download a pre-built ONNX model
import urllib.request
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Download MNIST model from ONNX Model Zoo
model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx"
output_path = "models/mnist.onnx"

print(f"Downloading MNIST model from ONNX Model Zoo...")
print(f"URL: {model_url}")
print(f"Output: {output_path}")

try:
    urllib.request.urlretrieve(model_url, output_path)
    print(f"\n✓ Successfully downloaded {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print("\nYou can now run SAMO with:")
    print(f"  python -m samo --model {output_path} --backend fpgaconvnet --platform platforms/zedboard.json --output-path outputs/ --optimiser rule")
except Exception as e:
    print(f"\n✗ Error downloading model: {e}")
    print("\nAlternative: You can manually download from:")
    print(f"  {model_url}")
    print(f"  And save it to: {output_path}")
