from src.inference.predict import Predictor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == '__main__':
    model =  AutoModelForSequenceClassification.from_pretrained('./final_model')
    tokenizer = AutoTokenizer.from_pretrained('./final_model')

    predictor = Predictor(model, tokenizer)

    # text = "The quick brown fox jumps over the lazy dog"
    # text = "The beautiful city of New York"
    # text = "That person is hardworking"
    # text = "The royal King sits on the throne of the Empire"
    text = "The royal King sits on the throne of the Empire in a small hall with few courtiers"

    result = predictor.predict(text)

    print(result)
    