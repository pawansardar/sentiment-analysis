from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalysisModel:
    def __init__(self, config):
        self.config = config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["name"],
            num_labels=int(self.config["num_classes"])
        )
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
