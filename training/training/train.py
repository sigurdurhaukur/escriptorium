import logging
import os

from kraken.configs import VGSLRecognitionTrainingConfig, VGSLRecognitionTrainingDataConfig
from kraken.models import convert_models
from kraken.train import VGSLRecognitionDataModule, VGSLRecognitionModel, KrakenTrainer

logger = logging.getLogger("training.train")


def run_training(
    train_arrow: str,
    val_arrow: str,
    output_path: str,
    batch_size: int = 4,
    max_epochs: int = 50,
    device: str = "mps",
    precision: str = "32",
    model_path: str | None = None,
    reorder: str = "L",
    num_workers: int = 0,
):
    data_config = VGSLRecognitionTrainingDataConfig(
        training_data=[train_arrow],
        evaluation_data=[val_arrow],
        format_type="binary",
        num_workers=num_workers,
    )
    train_config = VGSLRecognitionTrainingConfig(
        batch_size=batch_size,
        load_hyper_parameters=True,
        resize="union",
        reorder=reorder,
    )

    dm = VGSLRecognitionDataModule(data_config)

    if model_path:
        model = VGSLRecognitionModel.load_from_weights(model_path, train_config)
        logger.info("Loaded existing model from %s", model_path)
    else:
        model = VGSLRecognitionModel(train_config)
        logger.info("Created new VGSL model (default architecture)")

    trainer = KrakenTrainer(
        accelerator=device,
        devices="auto",
        precision=precision,
        max_epochs=max_epochs,
        enable_summary=False,
        enable_progress_bar=True,
        val_check_interval=1.0,
    )

    trainer.fit(model, dm)

    best_path = getattr(getattr(trainer, "checkpoint_callback", None), "best_model_path", None)
    if best_path and os.path.exists(best_path):
        logger.info("Converting best checkpoint to %s", output_path)
        convert_models([best_path], output_path)
    else:
        logger.warning("No best checkpoint found, saving final weights")
        final_ckpt = os.path.join(os.path.dirname(output_path), "final.ckpt")
        trainer.save_checkpoint(final_ckpt)
        convert_models([final_ckpt], output_path)

    logger.info("Model saved to %s", output_path)
