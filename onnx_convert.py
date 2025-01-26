from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# model_checkpoint = "finetuned-model"
# save_directory = "onnx/"

# ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# ort_model.save_pretrained(save_directory)
# tokenizer.save_pretrained(save_directory)

# print(f"ONNX model saved in {save_directory}")

onnx_model = ORTModelForSequenceClassification.from_pretrained("onnx/", file_name='model_quantized.onnx')
tokenizer = AutoTokenizer.from_pretrained("onnx/")

# Perform inference
text = "Tesla stocks dropped 42% while [TGT] rallied."

inputs = tokenizer(text, return_tensors="pt")

outputs = onnx_model(**inputs)
predicted_class = int(outputs.logits.argmax())
print(f"{predicted_class}")