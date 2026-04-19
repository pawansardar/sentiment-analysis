from src.inference.predictor import Predictor
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config.config_loader import load_config

if __name__ == '__main__':
    config = load_config()

    model_path = config["paths"]["model_dir"]

    model =  AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    predictor = Predictor(model, tokenizer)

    # text = "The quick brown fox jumps over the lazy dog"
    # text = "The beautiful city of New York"
    # text = "That person is hardworking"
    # text = "The royal King sits on the throne of the Empire"
    text = "The royal King sits on the throne of the Empire in a small hall with few courtiers"

    result = predictor.predict(text)

    print(result)
