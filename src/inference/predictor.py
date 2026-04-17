import torch

class Predictor:
    def __init__(self, model, tokenizer, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model.eval()
    
    def predict(self, text):
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits  # shape: [batch_size, num_classes]

        probs = torch.softmax(logits, dim=1)  # convert to probabilities

        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

        label = "Positive" if pred_class == 1 else "Negative"

        return {
            "label": label,
            "confidence": confidence
        }
