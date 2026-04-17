from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalysisModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
