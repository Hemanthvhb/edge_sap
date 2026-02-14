import torch
import torchvision.models as models

# 1. Load pretrained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# 2. Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export to ONNX
# We use opset_version=18 to match your newer PyTorch version
print("Exporting model...")
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18 
)

print("ONNX model exported successfully to model.onnx")