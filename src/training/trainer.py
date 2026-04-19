from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from src.utils.metrics import compute_metrics

def get_trainer(model, train_dataset, val_dataset, tokenizer, config):
    training_cfg = config["training"]

    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],

        eval_strategy="steps",
        eval_steps=500,

        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,

        logging_strategy="steps",
        logging_steps=100,

        learning_rate=float(training_cfg["learning_rate"]),
        per_device_train_batch_size=int(training_cfg["batch_size"]),
        num_train_epochs=int(training_cfg["num_epochs"]),
        
        weight_decay=float(training_cfg["weight_decay"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),

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
