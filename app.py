import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

# Load your fine-tuned model (replace with your model path if local)
import os

model_path = os.path.join(os.path.dirname(__file__), "best_model_hf")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)





# Prediction function
def predict_fake_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["Fake", "Real"]  # change order if your training was reversed
    return {labels[i]: float(probs[0][i]) for i in range(len(labels))}

# Gradio Interface
demo = gr.Interface(
    fn=predict_fake_news,
    inputs=gr.Textbox(lines=4, placeholder="Enter a news headline or article..."),
    outputs=gr.Label(num_top_classes=2),
    title="Fake News Detection with BERT",
    description="Enter a news article or headline, and the model will predict whether it's Real or Fake."
)

demo.launch()