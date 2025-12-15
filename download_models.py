"""
Download pre-built ONNX models that work well with SAMO
These models are from the ONNX Model Zoo and are similar to the ones in generate_networks.py
"""

import urllib.request
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Models available from ONNX Model Zoo
models_to_download = {
    'mnist': {
        'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx',
        'description': 'MNIST digit classifier (like LeNet architecture)',
        'size': 'Small (~26KB)'
    },
    'mobilenetv2': {
        'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx',
        'description': 'MobileNetV2 for ImageNet classification',
        'size': 'Medium (~14MB)'
    },
    'resnet18': {
        'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx',
        'description': 'ResNet-18 for ImageNet classification',
        'size': 'Medium (~44MB)'
    },
    'squeezenet': {
        'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx',
        'description': 'SqueezeNet 1.0 - lightweight CNN',
        'size': 'Small (~5MB)'
    },
}

print("=" * 70)
print("SAMO Model Downloader")
print("=" * 70)
print("\nAvailable models:")
for i, (name, info) in enumerate(models_to_download.items(), 1):
    print(f"\n{i}. {name}")
    print(f"   Description: {info['description']}")
    print(f"   Size: {info['size']}")

print("\n" + "=" * 70)
choice = input("\nWhich models do you want to download? (e.g., '1,2' or 'all'): ").strip().lower()

if choice == 'all':
    selected = list(models_to_download.keys())
else:
    try:
        indices = [int(x.strip()) for x in choice.split(',')]
        selected = [list(models_to_download.keys())[i-1] for i in indices]
    except:
        print("Invalid selection. Downloading only mnist as default.")
        selected = ['mnist']

print("\n" + "=" * 70)
print("Downloading selected models...")
print("=" * 70)

for name in selected:
    if name not in models_to_download:
        continue
    
    info = models_to_download[name]
    url = info['url']
    filename = url.split('/')[-1]
    output_path = os.path.join("models", filename)
    
    print(f"\n📥 Downloading {name}...")
    print(f"   URL: {url}")
    print(f"   Saving to: {output_path}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        file_size = os.path.getsize(output_path) / 1024
        if file_size > 1024:
            size_str = f"{file_size/1024:.1f} MB"
        else:
            size_str = f"{file_size:.1f} KB"
        print(f"   ✓ Success! ({size_str})")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

print("\n" + "=" * 70)
print("Download complete!")
print("=" * 70)

# List all downloaded models
print("\nModels in 'models/' directory:")
for f in sorted(os.listdir("models")):
    if f.endswith('.onnx'):
        path = os.path.join("models", f)
        size = os.path.getsize(path) / 1024
        if size > 1024:
            size_str = f"{size/1024:.2f} MB"
        else:
            size_str = f"{size:.1f} KB"
        print(f"  ✓ {f} ({size_str})")

print("\n" + "=" * 70)
print("Example SAMO commands:")
print("=" * 70)
print("\n1. Fast test with MNIST (rule-based):")
print("   python -m samo --model models/mnist-8.onnx --backend fpgaconvnet \\")
print("       --platform platforms/zedboard.json --output-path outputs/ \\")
print("       --optimiser rule --objective latency")

print("\n2. Optimize MobileNetV2 (annealing):")
print("   python -m samo --model models/mobilenetv2-12.onnx --backend fpgaconvnet \\")
print("       --platform platforms/zcu106.json --output-path outputs/ \\")
print("       --optimiser annealing --objective throughput --batch-size 256")

print("\n3. ResNet-18 latency optimization:")
print("   python -m samo --model models/resnet18-v2-7.onnx --backend fpgaconvnet \\")
print("       --platform platforms/u250_1slr.json --output-path outputs/ \\")
print("       --optimiser annealing --objective latency")

print("\n" + "=" * 70)
