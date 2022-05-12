import logging

from convolve.distortion import TextDistorter
from convolve.feature.projection import ProjectionEncoder
from convolve.feature import OrthographicForm

from optimizer import OptimizerConfig
from prnn_crf import PRNNCRFModel
from trainer import PRNNCRFTrainer
from utils import get_ner_rules

# Data
TRAIN_FILE = "/Users/seanmiller/dev/ner_data/conll2003/train.csv"
VAL_FILE = "/Users/seanmiller/dev/ner_data/conll2003/valid.csv"
TEST_FILE = "/Users/seanmiller/dev/ner_data/conll2003/test.csv"

# Model storage
MODEL_DIR = "/Users/seanmiller/dev/ner_pytorch"
EPOCHS = 1


def main():
    # Features
    feature_dim = 128
    max_seq_len = 128
    distort = 0.01
    orth = [
        OrthographicForm.all_caps,
        OrthographicForm.capitalized,
        OrthographicForm.is_numeric,
        OrthographicForm.has_numeric,
        OrthographicForm.is_punctuation,
    ]
    encoder = ProjectionEncoder(feature_dim, max_seq_len, orth)
    distorter = TextDistorter(distort)

    # Model
    bottleneck_dim = 32
    hidden_dim = 128
    num_rnn_layers = 1
    dropout = 0.4
    batch_size = 128

    ner_trainer = PRNNCRFTrainer(
        encoder=encoder, distorter=distorter, batch_size=batch_size
    )

    train_loader = ner_trainer.get_data_loader(TRAIN_FILE, train=True, shuffle=True)
    val_loader = ner_trainer.get_data_loader(VAL_FILE, train=False, shuffle=False)

    num_classes = ner_trainer.num_classes()

    crf_rules = get_ner_rules(ner_trainer.tag_to_idx)
    model_settings = {
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "fc_dim": bottleneck_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_rnn_layers,
        "dropout": dropout,
        "crf_rules": crf_rules,
    }

    optimizer = OptimizerConfig.Adam
    optimizer.set_lr(0.01)
    optimizer.set_betas((0.9, 0.999))
    optimizer.set_eps(1e-6)
    optimizer.set_decay(1e-6)
    optimizer.set_amsgrad(False)

    model = ner_trainer.train(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        save=MODEL_DIR,
        optimizer=optimizer,
        **model_settings,
    )

    print("Loading best model from training")

    model = PRNNCRFModel.load_from_dir(MODEL_DIR)
    test_loader = ner_trainer.get_data_loader(TEST_FILE, train=False, shuffle=False)
    ner_trainer.evaluate(model, test_loader)


# for reproducibility


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    main()
