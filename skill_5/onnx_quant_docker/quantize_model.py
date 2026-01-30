from onnxruntime.quantization import quantize_dynamic, QuantType

fp32_model = "model_fp32.onnx"
int8_model = "model_int8.onnx"

quantize_dynamic(
    model_input=fp32_model,
    model_output=int8_model,
    weight_type=QuantType.QInt8
)

print("Quantization completed!")
print("INT8 model saved as model_int8.onnx")
