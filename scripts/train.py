from src.models.model import SentimentAnalysisModel
from src.datasets.dataset import SentimentDataset
from src.training.trainer import get_trainer
from src.config.config_loader import load_config

if __name__ == '__main__':
    config = load_config()

    dataset = SentimentDataset(config)
    splits = dataset.get_dataset_split()

    model_obj = SentimentAnalysisModel(config)

    trainer = get_trainer(
        model=model_obj.get_model(),
        train_dataset=splits["train"],
        val_dataset=splits["validation"],
        tokenizer=model_obj.get_tokenizer(),
        config=config
    )

    trainer.train()
    
    final_model_path = config["paths"]["model_dir"]

    trainer.save_model(final_model_path)
    model_obj.get_tokenizer().save_pretrained(final_model_path)
