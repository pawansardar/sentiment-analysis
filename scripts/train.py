from src.models.model import SentimentAnalysisModel
from src.datasets.dataset import SentimentDataset
from src.training.trainer import get_trainer

if __name__ == '__main__':
    dataset = SentimentDataset()
    splits = dataset.get_dataset_split()

    model_obj = SentimentAnalysisModel()
    model = model_obj.get_model()

    trainer = get_trainer(
        model=model,
        train_dataset=splits["train"],
        val_dataset=splits["validation"],
        tokenizer=model_obj.get_tokenizer()
    )

    trainer.train()

    trainer.save_model("./final_model")
    model_obj.get_tokenizer().save_pretrained("./final_model")
