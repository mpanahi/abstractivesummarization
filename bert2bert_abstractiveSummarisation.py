import numpy as np
import pandas as pd
from IPython import display
from datasets import load_dataset


data = load_dataset("pn_summary")
train = data["train"]
print(len(train))
indices = np.random.randint(0, len(train), 10000).tolist()
#df = pd.DataFrame(train[indices])
df = pd.DataFrame(train)
df = df[["id", "title", "article", "summary", "category", "categories", "network", "link"]]
df["article"] = df["article"].apply(lambda t: t.replace('[n]', ' ')[:512] + ' [...]')
df["category"] = df["category"].apply(lambda t: train.features["category"].int2str(t))
df["network"] = df["network"].apply(lambda t: train.features["network"].int2str(t))
train_data=[]
for i in range(len(df)):
  dc={}
  dc["article_original"]=df["article"][i]
  dc["abstractive"]=df["summary"][i]
  train_data.append(dc)
import json

from torch.utils.data import DataLoader
#train_data = open("drive/MyDrive/train.jsonl", "r").read().splitlines()
def load_dataset(batch_size):
    #train_data = open("drive/MyDrive/train.jsonl", "r").read().splitlines()

    train_set = []
    for data in train_data:
        #data = json.loads(data)
        article_original = data["article_original"].replace("\n", " ")
        #article_original = [a.replace("\n", " ") for a in article_original]
        #article_original = " ".join(article_original)
        abstractive = data["abstractive"].replace("\n", " ")
        train_set.append((article_original, abstractive))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
    )

    return train_loader
from  transformers import AutoTokenizer
import torch
tokenizer=torch.load("tokenizer_bert")
#tokenizer=AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
#tokenizer=AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base",proxies={'http':'http://arvwtanr-rotate:8ttxmkq7qeuc@209.127.184.202:80','https':'http://arvwtanr-rotate:8ttxmkq7qeuc@209.127.184.202:80'})
import pytorch_lightning as pl
import torch

from pytorch_lightning import Trainer
from torch.optim import AdamW
from torch.utils.data import DataLoader


class LightningBase(pl.LightningModule):
    def __init__(
            self,
            model_save_path: str,
            max_len: int,
            batch_size: int,
            num_gpus: int,
            max_epochs:int=5,
            min_epochs:int=1,
            lr: float = 3e-5,
            weight_decay: float = 1e-4,
            save_step_interval: int = 1000,
            accelerator: str = "dp",
            precision: int = 16,
            use_amp: bool = True,
    ) -> None:
        """constructor of LightningBase"""

        super().__init__()
        self.model_save_path = model_save_path
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.max_len = max_len
        self.num_gpus = num_gpus
        self.max_epochs=5,
        self.min_epochs=1,
        self.save_step_interval = save_step_interval
        self.accelerator = accelerator
        self.precision = precision
        self.use_amp = use_amp
        self.model = None
        

    def configure_optimizers(self):
        """configure optimizers and lr schedulers"""
        no_decay = ["bias", "LayerNorm.weight"]
        model = self.model
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            params=optimizer_grouped_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        return [self.optimizer]

    def fit(self, train_dataloader: DataLoader):
        trainer = Trainer(
            gpus=self.num_gpus,
            distributed_backend=self.accelerator,
            precision=self.precision,
            max_epochs=self.max_epochs,
            min_epochs=self.min_epochs,
            amp_backend="apex" if self.use_amp else None,
        )

        trainer.fit(
            model=self, train_dataloader=train_dataloader,
        )

    def save_model(self) -> None:
        if (
                self.trainer.global_rank == 0
                and self.global_step % self.save_step_interval == 0
        ):
            torch.save(
                self.model.state_dict(),
                self.model_save_path + "." + str(self.global_step),
            )

from typing import List
from transformers import (
    EncoderDecoderModel,
    BertConfig,
    EncoderDecoderConfig,
    BertModel, BertTokenizer,
)

#from transformers.modeling_bart import shift_tokens_right
from kobert_transformers import get_tokenizer
import pytorch_lightning
import torch
from  transformers import AutoTokenizer
#from lightning_base import LightningBase


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens
class Bert2Bert(LightningBase):
    def __init__(
            self,
            model_save_path: str,
            batch_size: int,
            num_gpus: int,
            max_epochs:int=4,
            min_epochs=1,
            max_len: int = 512,
            lr: float = 3e-5,
            weight_decay: float = 1e-4,
            save_step_interval: int = 1000,
            accelerator: str = "dp",
            precision: int = 16,
            use_amp: bool = True,
    ) -> None:
        super(Bert2Bert, self).__init__(
            model_save_path=model_save_path,
            max_len=max_len,
            max_epochs=4,
            min_epochs=1,
            batch_size=batch_size,
            num_gpus=num_gpus,
            lr=lr,
            weight_decay=weight_decay,
            save_step_interval=save_step_interval,
           use_amp=use_amp,
            precision=precision
            
        )
        encoder_config = torch.load("encoder_config")
        decoder_config = torch.load("encoder_config")
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config, decoder_config
        )

        self.model = EncoderDecoderModel(config)
        self.tokenizer = KoBertTokenizer()
        #self.tokenizer =AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
        state_dict = torch.load("bert-fa-zwnj-base").state_dict()
        self.model.encoder.load_state_dict(state_dict)
        self.model.decoder.bert.load_state_dict(state_dict, strict=False)
        # cross attention이랑 lm head는 처음부터 학습


    def training_step(self, batch, batch_idx):
        src, tgt = batch[0], batch[1]
        print(src)
        print(tgt)
        #text_tokenized=tokenizer(text,return_tensors="pt").to(device)
 #print(len(text_tokenized['input_ids']))
 

        src_input = self.tokenizer.encode_batch(src, max_length=self.max_len)
        #src_input = self.tokenizer(src,return_tensors="pt")
        tgt_input = self.tokenizer.encode_batch(tgt, max_length=self.max_len)
        #tgt_input = self.tokenizer(tgt,return_tensors="pt")
        input_ids = src_input["input_ids"].to(self.device)
        attention_mask = src_input["attention_mask"].to(self.device)
        labels = tgt_input["input_ids"].to(self.device)
        decoder_input_ids = shift_tokens_right(
            labels, self.tokenizer.token2idx("[PAD]")
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

        lm_logits = outputs[0]
        loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.token2idx("[PAD]")
        )

        lm_loss = loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        self.save_model()
        return {"loss": lm_loss}


class KoBertTokenizer(object):
    def __init__(self):
        self.tokenizer = torch.load("tokenizer_bert")
        self.token2idx = self.tokenizer.convert_tokens_to_ids
        self.idx2token = self.tokenizer.convert_ids_to_tokens

    def encode_batch(self, x: List[str], max_length):
        max_len = 0
        result_tokenization = []

        for i in x:
            tokens = self.tokenizer.encode(i, max_length=max_length, truncation=True)
            result_tokenization.append(tokens)

            if len(tokens) > max_len:
                max_len = len(tokens)

        padded_tokens = []
        for tokens in result_tokenization:
            padding = (torch.ones(max_len) * self.token2idx("[PAD]")).long()
            padding[: len(tokens)] = torch.tensor(tokens).long()
            padded_tokens.append(padding.unsqueeze(0))

        padded_tokens = torch.cat(padded_tokens, dim=0).long()
        mask_tensor = torch.ones(padded_tokens.size()).long()

        attention_mask = torch.where(
            padded_tokens == self.token2idx("[PAD]"), padded_tokens, mask_tensor * -1
        ).long()
        attention_mask = torch.where(
            attention_mask == -1, attention_mask, mask_tensor * 0
        ).long()
        attention_mask = torch.where(
            attention_mask != -1, attention_mask, mask_tensor
        ).long()

        return {
            "input_ids": padded_tokens.long(),
            "attention_mask": attention_mask.long(),
        }

    def decode(self, tokens):
        # remove special tokens
        # unk, pad, cls, sep, mask
        tokens = [token for token in tokens
                  if token not in [0, 1, 2, 3, 4]]

        decoded = [self.idx2token(token) for token in tokens]
        if "▁" in decoded[0] and "▁" in decoded[1]:
            # fix decoding bugs
            tokens = tokens[1:]

        return self.tokenizer.decode(tokens)

    def decode_batch(self, list_of_tokens):
        return [self.decode(tokens) for tokens in list_of_tokens]



#print(torch.version.cuda)
import torch
torch.cuda.empty_cache()
#if __name__ == '__main__':
trainer = Bert2Bert(
        model_save_path="model_summary_persian_new.pt",
        batch_size=8,
        num_gpus=1,
        max_epochs=5,
        min_epochs=1
    )

train = load_dataset(batch_size=trainer.batch_size)
print("****")
#print((train.dataset))
print("****")
trainer.fit(train)
