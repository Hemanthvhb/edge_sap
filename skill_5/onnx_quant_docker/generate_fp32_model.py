import torch
import torch.nn as nn

# Simple FP32 model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
model.eval()

dummy_input = torch.randn(1, 10)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model_fp32.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)

print("FP32 ONNX model generated: model_fp32.onnx")
