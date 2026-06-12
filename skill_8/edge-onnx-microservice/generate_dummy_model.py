import torch
import torch.nn as nn

# Define a simple model: a single linear layer
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(4, 1) # Expects input vector of size 4 to match user test command

    def forward(self, x):
        return self.linear(x)

model = DummyModel()
model.eval()

# Create dummy input (batch_size=1, features=4)
dummy_input = torch.randn(1, 4)

# Export the model
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx", 
    export_params=True, 
    opset_version=11, 
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Dummy model.onnx updated successfully with input size [batch_size, 4]")
