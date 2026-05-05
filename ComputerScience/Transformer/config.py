from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def get_config():
  return dict(
    batch_size = 32,
    num_epochs = 20,
    lr = 1e-4,

    seq_len = 350,
    embedding_dim = 256,
    num_layers = 3,
    num_heads = 8,
    dropout = 0.1,
    hidden_dim = 1024,

    lang_src = "english",
    lang_tgt = "french",

    model_folder = "weights",
    model_save_path = "tmodel_",
    preload = None,
    tokenizer_file = str(BASE_DIR / "{lang}_tokenizer_path"),
    experiment_name = str( BASE_DIR / "runs/tmodel_experiment")
  )

def get_weights_file_path(config, epoch: str):
  model_folder = config["model_folder"]
  model_basename = config["model_save_path"]
  model_filename = f"{model_basename}{epoch}.pt"
  return str(BASE_DIR / model_folder / model_filename)