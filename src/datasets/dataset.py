from datasets import load_dataset
from transformers import AutoTokenizer

class SentimentDataset:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.dataset = load_dataset("glue", "sst2")

        self.dataset = self.preprocess(self.dataset)

    def preprocess(self, dataset):
        def tokenize(example):
            text_key = "sentence" if "sentence" in example else "text"
            return self.tokenizer(example[text_key], truncation=True)
        
        dataset = dataset.map(tokenize, batched=True)
        
        dataset = dataset.rename_column("label", "labels")

        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        return dataset
    
    def get_dataset_split(self):
        return self.dataset
