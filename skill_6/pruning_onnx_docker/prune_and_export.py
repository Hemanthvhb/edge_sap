import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import onnx

torch.set_printoptions(sci_mode=False, precision=4)

# -----------------------------
# 1. DEFINE MODEL
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 24 * 24, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model = SimpleCNN()
model.eval()

# -----------------------------
# 2. BEFORE PRUNING (BASELINE)
# -----------------------------
print("\n===== BEFORE PRUNING =====")

print("Conv1 weight shape:", model.conv1.weight.shape)
print("Conv1 non-zero weights:",
      torch.count_nonzero(model.conv1.weight).item())

print("\nSample Conv1 weights (first filter):")
print(model.conv1.weight[0])

# -----------------------------
# 3. UNSTRUCTURED PRUNING
# -----------------------------
print("\n===== APPLYING UNSTRUCTURED PRUNING =====")

prune.l1_unstructured(
    model.conv1,
    name="weight",
    amount=0.4
)

# Before removing mask
print("Conv1 weight after masking (sparse):")
print(model.conv1.weight[0])

# Make pruning permanent
prune.remove(model.conv1, "weight")

print("\n===== AFTER UNSTRUCTURED PRUNING =====")
print("Conv1 non-zero weights:",
      torch.count_nonzero(model.conv1.weight).item())

print("Conv1 shape remains SAME:",
      model.conv1.weight.shape)

print("\nSample Conv1 weights after pruning:")
print(model.conv1.weight[0])

# -----------------------------
# 4. STRUCTURED PRUNING
# -----------------------------
print("\n===== APPLYING STRUCTURED PRUNING =====")

prune.ln_structured(
    model.conv2,
    name="weight",
    amount=0.25,
    n=2,
    dim=0
)

# Make pruning permanent
prune.remove(model.conv2, "weight")

print("\n===== AFTER STRUCTURED PRUNING =====")

print("Conv2 weight shape:",
      model.conv2.weight.shape)

# Check zeroed-out channels
zero_channels = []
for i in range(model.conv2.weight.shape[0]):
    if torch.count_nonzero(model.conv2.weight[i]) == 0:
        zero_channels.append(i)

print("Completely pruned channels in Conv2:", zero_channels)

# -----------------------------
# 5. PARAMETER COUNT COMPARISON
# -----------------------------
def count_nonzero_params(model):
    return sum(torch.count_nonzero(p).item() for p in model.parameters())

print("\n===== PARAMETER COMPARISON =====")
print("Total non-zero parameters after pruning:",
      count_nonzero_params(model))

# -----------------------------
# 6. EXPORT TO ONNX
# -----------------------------
dummy_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy_input,
    "pruned_model.onnx",
    export_params=True,
    opset_version=13,
    input_names=["input"],
    output_names=["output"]
)

print("\nONNX model exported successfully")

# -----------------------------
# 7. VALIDATE ONNX
# -----------------------------
onnx_model = onnx.load("pruned_ model.onnx")
onnx.checker.check_model(onnx_model)

print("ONNX model validation PASSED")
