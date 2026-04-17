from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from src.utils.metrics import compute_metrics

def get_trainer(model, train_dataset, val_dataset, tokenizer):
    training_args = TrainingArguments(
        output_dir="./experiments",

        eval_strategy="steps",
        eval_steps=500,

        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,

        logging_strategy="steps",
        logging_steps=100,

        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        
        weight_decay=0.01,
        warmup_ratio=0.1,

        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # padding

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    return trainer
