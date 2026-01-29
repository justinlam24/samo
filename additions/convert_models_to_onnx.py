"""
Helper script to convert the 12 modern CNN models to ONNX format.

This script downloads pretrained models from PyTorch/TensorFlow and converts them to ONNX.
Install required packages first:
    pip install torch torchvision onnx segmentation-models-pytorch timm

Usage: python convert_models_to_onnx.py
"""

import torch
import torch.nn as nn
import onnx
import onnx.helper
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional imports - will be checked in try/except blocks
try:
    import segmentation_models_pytorch as smp  # type: ignore
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


def add_kernel_shape_to_convs(model_path):
    """
    Post-process ONNX model to add kernel_shape attribute to Conv nodes.
    fpgaconvnet parser requires explicit kernel_shape attributes.
    """
    model = onnx.load(model_path)
    graph = model.graph
    
    # Build a map of initializer names to their shapes
    initializer_shapes = {}
    for init in graph.initializer:
        initializer_shapes[init.name] = list(init.dims)
    
    for node in graph.node:
        if node.op_type == 'Conv':
            # Check if kernel_shape already exists
            has_kernel_shape = any(attr.name == 'kernel_shape' for attr in node.attribute)
            
            if not has_kernel_shape:
                # Get kernel shape from weights (second input)
                if len(node.input) >= 2:
                    weight_name = node.input[1]
                    if weight_name in initializer_shapes:
                        weight_shape = initializer_shapes[weight_name]
                        # Weight shape is [out_channels, in_channels/groups, kH, kW]
                        if len(weight_shape) >= 4:
                            kernel_shape = weight_shape[2:]  # [kH, kW]
                            attr = onnx.helper.make_attribute('kernel_shape', kernel_shape)
                            node.attribute.append(attr)
    
    onnx.save(model, model_path)


def export_to_onnx(model, input_shape, output_path, model_name):
    """
    Export a PyTorch model to ONNX format compatible with fpgaconvnet.
    
    Args:
        model: PyTorch model
        input_shape: Tuple of (batch, channels, height, width)
        output_path: Path to save ONNX file
        model_name: Name for logging
    
    Note: Uses opset 9 for fpgaconvnet compatibility and adds kernel_shape attrs.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    print(f"Converting {model_name}...", end=" ")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=9,  # Use opset 9 for fpgaconvnet compatibility
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # Remove dynamic axes - fpgaconvnet expects fixed batch size
        )
        
        # Post-process to ensure kernel_shape attributes are present
        add_kernel_shape_to_convs(str(output_path))
        
        # Verify the model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✓ Saved to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def create_models():
    """
    Create and export 12 fpgaconvnet-compatible CNN models.
    
    NOTE: fpgaconvnet only supports sequential architectures without skip connections.
    These models are designed to represent different CNN patterns while being compatible.
    """
    
    models_dir = Path("models/new")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # ============================================================================
    # Image Classification - Various depths/widths (VGG-style sequential)
    # ============================================================================
    
    # 1. Deep narrow network (VGG-style)
    try:
        class DeepNarrowNet(nn.Module):
            """Deep network with narrow channels - 16 conv layers"""
            def __init__(self, num_classes=1000):
                super().__init__()
                self.features = nn.Sequential(
                    # Block 1
                    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    # Block 2
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    # Block 3
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    # Block 4
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    # Block 5
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512 * 7 * 7, 4096), nn.ReLU(),
                    nn.Linear(4096, 4096), nn.ReLU(),
                    nn.Linear(4096, num_classes),
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = DeepNarrowNet()
        success = export_to_onnx(model, (1, 3, 224, 224), models_dir / "deep_narrow.onnx", "DeepNarrow")
        results.append(("DeepNarrow (VGG-16 style)", success))
    except Exception as e:
        print(f"DeepNarrow failed: {e}")
        results.append(("DeepNarrow", False))
    
    # 2. Wide shallow network
    try:
        class WideShallowNet(nn.Module):
            """Shallow network with wide channels - 8 conv layers"""
            def __init__(self, num_classes=1000):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 128, 7, stride=2, padding=3), nn.ReLU(),
                    nn.MaxPool2d(3, 2, padding=1),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(1024, 2048, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(2048, 2048, 3, padding=1), nn.ReLU(),
                    nn.AvgPool2d(7),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(2048, num_classes),
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = WideShallowNet()
        success = export_to_onnx(model, (1, 3, 224, 224), models_dir / "wide_shallow.onnx", "WideShallow")
        results.append(("WideShallow", success))
    except Exception as e:
        print(f"WideShallow failed: {e}")
        results.append(("WideShallow", False))
    
    # ============================================================================
    # Object Detection Backbones (Sequential feature extractors)
    # ============================================================================
    
    # 3. Detection backbone - multi-scale features
    try:
        class DetectionBackbone(nn.Module):
            """Feature pyramid-like backbone (sequential)"""
            def __init__(self):
                super().__init__()
                self.stage1 = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                )
                self.stage2 = nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                )
                self.stage3 = nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                )
                self.stage4 = nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                )
                self.head = nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(512, 255, 1),  # Detection head (80 classes + 5) * 3 anchors
                )
            
            def forward(self, x):
                x = self.stage1(x)
                x = self.stage2(x)
                x = self.stage3(x)
                x = self.stage4(x)
                x = self.head(x)
                return x
        
        model = DetectionBackbone()
        success = export_to_onnx(model, (1, 3, 640, 640), models_dir / "detection_backbone.onnx", "DetectionBackbone")
        results.append(("DetectionBackbone", success))
    except Exception as e:
        print(f"DetectionBackbone failed: {e}")
        results.append(("DetectionBackbone", False))
    
    # 4. MobileNet-style depthwise separable (sequential)
    try:
        class DepthwiseSeparableConv(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.conv = nn.Sequential(
                    # Depthwise
                    nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch),
                    nn.ReLU(),
                    # Pointwise
                    nn.Conv2d(in_ch, out_ch, 1),
                    nn.ReLU(),
                )
            def forward(self, x):
                return self.conv(x)
        
        class MobileStyleNet(nn.Module):
            """MobileNet-style architecture without skip connections"""
            def __init__(self, num_classes=1000):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                    DepthwiseSeparableConv(32, 64),
                    DepthwiseSeparableConv(64, 128, stride=2),
                    DepthwiseSeparableConv(128, 128),
                    DepthwiseSeparableConv(128, 256, stride=2),
                    DepthwiseSeparableConv(256, 256),
                    DepthwiseSeparableConv(256, 512, stride=2),
                    DepthwiseSeparableConv(512, 512),
                    DepthwiseSeparableConv(512, 512),
                    DepthwiseSeparableConv(512, 512),
                    DepthwiseSeparableConv(512, 512),
                    DepthwiseSeparableConv(512, 512),
                    DepthwiseSeparableConv(512, 1024, stride=2),
                    DepthwiseSeparableConv(1024, 1024),
                    nn.AvgPool2d(7),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(1024, num_classes),
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = MobileStyleNet()
        success = export_to_onnx(model, (1, 3, 224, 224), models_dir / "mobile_style.onnx", "MobileStyle")
        results.append(("MobileStyle (depthwise separable)", success))
    except Exception as e:
        print(f"MobileStyle failed: {e}")
        results.append(("MobileStyle", False))
    
    # ============================================================================
    # Semantic Segmentation (Encoder-only, sequential)
    # ============================================================================
    
    # 5. Segmentation encoder (FCN-style)
    try:
        class SegmentationEncoder(nn.Module):
            """FCN-style segmentation encoder"""
            def __init__(self, num_classes=21):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    # Score layer
                    nn.Conv2d(512, num_classes, 1),
                )
            
            def forward(self, x):
                return self.encoder(x)
        
        model = SegmentationEncoder()
        success = export_to_onnx(model, (1, 3, 512, 512), models_dir / "segmentation_encoder.onnx", "SegmentationEncoder")
        results.append(("SegmentationEncoder (FCN-style)", success))
    except Exception as e:
        print(f"SegmentationEncoder failed: {e}")
        results.append(("SegmentationEncoder", False))
    
    # 6. Dense segmentation network
    try:
        class DenseSegNet(nn.Module):
            """Densely connected segmentation (sequential approximation)"""
            def __init__(self, num_classes=21):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 48, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(192, 192, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(192, 192, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(192, num_classes, 1),
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = DenseSegNet()
        success = export_to_onnx(model, (1, 3, 256, 256), models_dir / "dense_seg.onnx", "DenseSegNet")
        results.append(("DenseSegNet", success))
    except Exception as e:
        print(f"DenseSegNet failed: {e}")
        results.append(("DenseSegNet", False))
    
    # ============================================================================
    # Super-Resolution (Sequential without skip connections)
    # ============================================================================
    
    # 7. SRCNN-style super resolution
    try:
        class SRCNN(nn.Module):
            """SRCNN - Simple super resolution CNN"""
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 64, 9, padding=4), nn.ReLU(),
                    nn.Conv2d(64, 32, 5, padding=2), nn.ReLU(),
                    nn.Conv2d(32, 3, 5, padding=2),
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = SRCNN()
        success = export_to_onnx(model, (1, 3, 128, 128), models_dir / "srcnn.onnx", "SRCNN")
        results.append(("SRCNN (super resolution)", success))
    except Exception as e:
        print(f"SRCNN failed: {e}")
        results.append(("SRCNN", False))
    
    # 8. VDSR-style deep SR (sequential)
    try:
        class VDSR(nn.Module):
            """VDSR-style deep super resolution (no skip)"""
            def __init__(self, num_layers=20):
                super().__init__()
                layers = [nn.Conv2d(3, 64, 3, padding=1), nn.ReLU()]
                for _ in range(num_layers - 2):
                    layers.extend([nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()])
                layers.append(nn.Conv2d(64, 3, 3, padding=1))
                self.net = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.net(x)
        
        model = VDSR()
        success = export_to_onnx(model, (1, 3, 128, 128), models_dir / "vdsr.onnx", "VDSR")
        results.append(("VDSR (deep SR)", success))
    except Exception as e:
        print(f"VDSR failed: {e}")
        results.append(("VDSR", False))
    
    # ============================================================================
    # Medical Imaging (2D variants for compatibility)
    # ============================================================================
    
    # 9. Medical imaging encoder
    try:
        class MedicalEncoder(nn.Module):
            """Medical imaging feature extractor"""
            def __init__(self, in_channels=1, num_classes=10):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                    nn.AvgPool2d(8),
                    nn.Flatten(),
                    nn.Linear(256, num_classes),
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = MedicalEncoder(in_channels=1, num_classes=10)
        success = export_to_onnx(model, (1, 1, 64, 64), models_dir / "medical_encoder.onnx", "MedicalEncoder")
        results.append(("MedicalEncoder (grayscale)", success))
    except Exception as e:
        print(f"MedicalEncoder failed: {e}")
        results.append(("MedicalEncoder", False))
    
    # 10. Multi-scale medical (dilated convolutions)
    try:
        class MultiScaleMedical(nn.Module):
            """Multi-scale receptive field for medical imaging"""
            def __init__(self, in_channels=1, num_classes=5):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 128, 3, padding=4, dilation=4), nn.ReLU(),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                    nn.AvgPool2d(16),
                    nn.Flatten(),
                    nn.Linear(256, num_classes),
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = MultiScaleMedical()
        success = export_to_onnx(model, (1, 1, 64, 64), models_dir / "multiscale_medical.onnx", "MultiScaleMedical")
        results.append(("MultiScaleMedical (dilated)", success))
    except Exception as e:
        print(f"MultiScaleMedical failed: {e}")
        results.append(("MultiScaleMedical", False))
    
    # ============================================================================
    # Embedding/Feature Networks (for self-supervised)
    # ============================================================================
    
    # 11. Feature embedding network
    try:
        class EmbeddingNet(nn.Module):
            """Feature embedding network for self-supervised learning"""
            def __init__(self, embedding_dim=128):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                    nn.AvgPool2d(7),
                    nn.Flatten(),
                    nn.Linear(512, embedding_dim),
                )
            
            def forward(self, x):
                return self.encoder(x)
        
        model = EmbeddingNet()
        success = export_to_onnx(model, (1, 3, 224, 224), models_dir / "embedding_net.onnx", "EmbeddingNet")
        results.append(("EmbeddingNet (self-supervised)", success))
    except Exception as e:
        print(f"EmbeddingNet failed: {e}")
        results.append(("EmbeddingNet", False))
    
    # 12. Compact classifier
    try:
        class CompactNet(nn.Module):
            """Compact network for edge deployment"""
            def __init__(self, num_classes=100):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                    nn.AvgPool2d(4),
                    nn.Flatten(),
                    nn.Linear(128, num_classes),
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = CompactNet()
        success = export_to_onnx(model, (1, 3, 32, 32), models_dir / "compact_net.onnx", "CompactNet")
        results.append(("CompactNet (edge deployment)", success))
    except Exception as e:
        print(f"CompactNet failed: {e}")
        results.append(("CompactNet", False))
    
    # ============================================================================
    # Summary
    # ============================================================================
    
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nSuccessfully converted: {successful}/{total} models")
    print(f"\nDetailed results:")
    for model_name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {model_name}")
    
    print(f"\n⚠ Note: These are fpgaconvnet-compatible sequential models.")
    print(f"  Modern architectures with skip connections (ResNet, U-Net, etc.)")
    print(f"  are not supported by fpgaconvnet.")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    print("Converting 12 fpgaconvnet-compatible CNN models to ONNX format...\n")
    create_models()
    print("Done! Models saved to models/new/ directory.")
    print("\nNext steps:")
    print("  1. Run benchmark: python additions/test_model_set.py --platform platforms/u250_1slr.json")

