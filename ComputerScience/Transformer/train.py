import warnings
from typing import Literal, cast

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader

import kagglehub
from kagglehub import KaggleDatasetAdapter

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import BilingualDataset
from model import buildTransformer
from config import get_config, get_weights_file_path

def get_all_sentences(df, lang: Literal["english", "french"]):
  for item in df:
    yield item['English words/sentences'] if lang == "english" else item['French words/sentences']

def get_or_build_tokeniser(config, df: Dataset, lang: Literal["english", "french"]):
  tokenizer_path = Path(config["tokenizer_file"].format(lang=lang))

  if not Path.exists(tokenizer_path):
    print(f"Building {lang} tokenizer...")
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(df, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    print(f"Loading {lang} tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  
  return tokenizer

def get_dataset(config):
  df: Dataset = kagglehub.load_dataset(KaggleDatasetAdapter.HUGGING_FACE, "devicharith/language-translation-englishfrench", "eng_-french.csv")
  
  tokenizer_src = get_or_build_tokeniser(config, df, "english")
  tokenizer_tgt = get_or_build_tokeniser(config, df, "french")

  train_df_size = int(len(df) * 0.8) #type: ignore
  val_df_size = len(df) - train_df_size #type: ignore

  train_df_raw, val_df_raw = random_split(df, [train_df_size, val_df_size])


  train_df = BilingualDataset(train_df_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
  val_df = BilingualDataset(val_df_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

  max_len_src = 0
  max_len_tgt = 0

  for item in df:
    src_len = len(tokenizer_src.encode(item['English words/sentences']).ids)
    tgt_len = len(tokenizer_tgt.encode(item['French words/sentences']).ids)

    max_len_src = max(max_len_src, src_len)
    max_len_tgt = max(max_len_tgt, tgt_len)
  
  print(f"Max sequence length for source language: {max_len_src}")
  print(f"Max sequence length for target language: {max_len_tgt}")

  train_dataloader = torch.utils.data.DataLoader(train_df, batch_size=config["batch_size"], shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(val_df, batch_size=1, shuffle=False)

  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_size_src: int, vocab_size_tgt: int):
  from model import buildTransformer

  model = buildTransformer(
    src_vocab_size=vocab_size_src,
    tgt_vocab_size=vocab_size_tgt,
    src_seq_len=config["seq_len"],
    tgt_seq_len=config["seq_len"],
    embedding_dim=config["embedding_dim"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    dropout=config["dropout"],
    hidden_dim=config["hidden_dim"]
  )

  return model

def train_model(config):
  #Define the device for tensors
  if torch.backends.mps.is_available():
      device = torch.device("mps")
  elif torch.cuda.is_available():
      device = torch.device("cuda")
  else:
      device = torch.device("cpu")
  print(f"Using device: {device}")
  
  torch.set_default_dtype(torch.float32)

  Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

  #Get the dataset and tokenizers
  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
  model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

  #Tensorboard: allows visualising the training process, including the loss and other metrics, in real time.
  writer = SummaryWriter(config["experiment_name"])

  optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

  initial_epoch = 0
  global_step = 0
  if config["preload"] is not None:
    model_filename = get_weights_file_path(config, config["preload"])
    print(f'Loading model weights from {model_filename}...')

    state = torch.load(model_filename)
    initial_epoch = state['epoch'] + 1

    optimizer.load_state_dict(state['optimizer'])
    global_step = state['global_step']

  loss_fn = nn.CrossEntropyLoss(ignore_index=cast(int, tokenizer_src.token_to_id("[PAD]")), label_smoothing= 0.1).to(device)

  for epoch in range(initial_epoch, config["num_epochs"]):
    model.train()
    batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

    for batch in batch_iterator:
      encoder_input = batch['encoder_input'].to(device)
      decoder_input = batch['decoder_input'].to(device)
      encoder_mask = batch['encoder_mask'].to(device)
      decoder_mask = batch['decoder_mask'].to(device)
      
      encoder_output = model.encode(encoder_input, encoder_mask)
      decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
      proj_output = model.project(decoder_output)

      label = batch['label'].to(device)
      loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
      batch_iterator.set_postfix({"loss": loss.item()})

      writer.add_scalar("train loss", loss.item(), global_step)
      writer.flush()

      loss.backward()

      optimizer.step()
      optimizer.zero_grad()

      global_step += 1

    # Save the model weights after each epoch
    model_filename = get_weights_file_path(config, f'{epoch:02d}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)

if __name__ == '__main__':
  warnings.filterwarnings('ignore')
  config = get_config()
  train_model(config)