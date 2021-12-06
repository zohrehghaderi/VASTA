from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from model.Mlpmax import MLP
import torch.nn as nn
import pytorch_lightning as pl
from typing import Union, List, Any, Callable, Optional
import torch
from dataloader.data_loader import Video_Caption_Loader
from torch.utils.data import DataLoader
from model.Encoder import swin_encoder
from model.Prepare_inputs_sos import SOSSwinBert
from transformers import BertTokenizer, BertConfig
from nlp_metrics.NLP_metrics import convert_list_to_string, nlp_metric_bert


PAD_token = 0
EOS_token = 2
SOS_token = 3

beam_num = 3


class Swin_BERT_Semantics(pl.LightningModule):

    def __init__(self, swin_freeze, in_size, lr, lambda_, hidden_sizes, out_size, gra_clip, drop_swin,
                 weight_decay, max_length, mlp_freeze, drop_mlp, drop_bert, bs, dataset, config_data, check_semantics,
                 checkpoint_encoder, using_adaptive):
        super(Swin_BERT_Semantics, self).__init__()
        #prepar dataset
        self.dataset = dataset
        self.path_data = self.dataset.Dataset_path
        self.config_data = config_data
        self.vocab = self.dataset.read_vocab()
        self.val = self.dataset.read_val()
        self.using_adaptive=using_adaptive

        #training parameter
        self.gradient_clip = gra_clip
        self.drop_bert = drop_bert
        self.drop_swin = drop_swin
        self.drop_mlp = drop_mlp
        self.weight_decay = weight_decay
        self.batch_size = bs
        self.lr = lr
        self.max_length = max_length
        self.loss_semantics = nn.BCEWithLogitsLoss()
        self.accuracy = nlp_metric_bert()
        self.lambda_ = lambda_

        #Swin-Encoder
        self.checkpoint_encoder = checkpoint_encoder
        self.encoder = swin_encoder(device=self.device, drop=self.drop_swin, checkpoint_encoder=self.checkpoint_encoder)
        self.encoder_change = nn.Linear(1024, 768)
        if swin_freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False


        #BERT-Decoder
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained("bert-base-uncased", bos_token_id=101, pad_token_id=0, eos_token_id=102)
        config.is_decoder = True
        config.add_cross_attention = True
        config.hidden_dropout_prob = drop_bert
        config.attention_probs_dropout_prob = drop_bert
        self.decoder = SOSSwinBert.from_pretrained('bert-base-uncased', config=config)

        #Semantics-Network
        self.net = MLP(in_size=in_size, hidden_sizes=hidden_sizes, out_size=out_size,
                       dropout_p=self.drop_mlp, have_last_bn=True)
        if check_semantics != '':
            checkpoint = torch.load(check_semantics, map_location=self.device)
            self.net.load_state_dict(checkpoint['state_dict'])
        if mlp_freeze:
            for param in self.net.parameters():
                param.requires_grad = False

    def _load_checkpoint(self, ch):
        checkpoint = torch.load(ch, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])

    def forward(self, enc_inputs, dec_inputs):
        self.encoder.eval()
        self.net.eval()
        self.decoder.eval()

        batch_size = enc_inputs.shape[0]
        #encode video
        enc_outputs = self.encoder(enc_inputs)
        enc_outputs_vision = self.encoder_change(enc_outputs)
        semantics = self.net(enc_outputs)

        #prepare SOS, ourput encoder to generate text
        seq_input = torch.zeros(batch_size, 1, dtype=torch.int, device=self.device)
        seq_input[:, 0] = self.decoder.config.bos_token_id
        expanded_return_idx = (torch.arange(seq_input.shape[0]).view(-1, 1).repeat(1, beam_num).view(-1).to(self.device))
        encoder_hidden_states = enc_outputs_vision.index_select(0, expanded_return_idx)
        mask_p = torch.ones(batch_size, 1, dtype=torch.int, device=self.device)
        semantics = semantics.index_select(0, expanded_return_idx)
        semantics = semantics.unsqueeze(1)
        model_kwargs = {"encoder_hidden_states": encoder_hidden_states, "inputs_embeds": semantics,"attention_mask": mask_p}

        #generate text
        outputs = self.decoder.generate(input_ids=seq_input,
                                        bos_token_id=self.decoder.config.bos_token_id,
                                        eos_token_id=self.decoder.config.eos_token_id,
                                        pad_token_id=self.decoder.config.pad_token_id,
                                        max_length=self.max_length,
                                        num_beams=beam_num,
                                        num_return_sequences=1, **model_kwargs)

        return outputs

    def training_step(self, train_batch, batch_idx):
        if self.global_step == 0:
            # only log hyperparameters once
            self.logger.experiment.add_text('learning rate', str(self.lr), self.global_step)
            self.logger.experiment.add_text('drop_bert', str(self.drop_bert), self.global_step)
            self.logger.experiment.add_text('drop_swin', str(self.drop_swin), self.global_step)
            self.logger.experiment.add_text('drop_mlp', str(self.drop_mlp), self.global_step)
            self.logger.experiment.add_text('gradient clip', str(self.gradient_clip), self.global_step)
            self.logger.experiment.add_text('weight decay', str(self.weight_decay), self.global_step)
            self.logger.experiment.add_text('batch size', str(self.batch_size), self.global_step)
            self.logger.experiment.add_text('max length', str(self.max_length), self.global_step)
            self.logger.experiment.add_text('lambda_', str(self.lambda_), self.global_step)

        enc_inputs, dec_inputs, tag = train_batch

        bs = enc_inputs.shape[0]

        #encode video
        enc_outputs = self.encoder(enc_inputs)
        enc_outputs_vision = self.encoder_change(enc_outputs)

        #make semantice-visual feature
        semantics = self.net(enc_outputs)
        loss_semantics = self.loss_semantics(semantics, tag)


        #convert token to string for bert-tokensier
        dec_inpt_bert = []
        inputarr = dec_inputs.cpu().detach().numpy()
        for di in range(0, bs):
            ignore = [SOS_token, EOS_token, PAD_token]
            sent = [word for word in inputarr[di] if word not in ignore]
            temp = []
            for iii in range(sent.__len__()):
                temp.append(self.vocab.index_to_token[sent[iii]])
            dec_inpt_bert.append(convert_list_to_string(temp))

        del dec_inputs
        del inputarr

        # tokenizer
        token_input = self.tokenizer(dec_inpt_bert, return_tensors="pt", padding=True)
        token_input = token_input.to(self.device)
        label = token_input["input_ids"] - (token_input["input_ids"] == 0) * 100
        input_embed = self.decoder.base_model.embeddings(token_input["input_ids"])

        # replace sos to sematnics
        for l in range(0, bs):
            input_embed[l][0][:] = semantics[l][:]

        # pass decoder
        outputs = self.decoder(inputs_embeds=input_embed, labels=label, encoder_hidden_states=enc_outputs_vision)
        loss = outputs.loss

        # only examples log every 250 steps
        if batch_idx % 250 == 0:
            # log some text examples to TB
            decoded_text = self.tokenizer.batch_decode(outputs.logits.argmax(dim=2))
            generated_text = ""
            gt = ""
            for id in range(0, self.batch_size):
                generated_text += str(id + 1) + ": " + decoded_text[id] + '\n'
                gt += str(id + 1) + ": " + dec_inpt_bert[id] + '\n'

            self.logger.experiment.add_text("train/generated_text", generated_text, self.global_step)
            self.logger.experiment.add_text("train/gt_text", gt, self.global_step)

        #loss
        final_loss = loss + self.lambda_ * loss_semantics
        self.log('loss/final_loss', final_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss/loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss/loss_semantics', loss_semantics, on_epoch=True, prog_bar=True, logger=True)

        return final_loss

    def validation_step(self, train_batch, batch_idx):
        self.encoder.eval()
        self.net.eval()
        self.decoder.eval()

        enc_inputs, dec_inputs, label = train_batch
        batch_size = enc_inputs.shape[0]

        #encode video
        enc_outputs = self.encoder(enc_inputs)
        enc_outputs_vision = self.encoder_change(enc_outputs)
        semantics = self.net(enc_outputs)

        #prepare SOS, output encoder to generate text
        seq_input = torch.zeros(batch_size, 1, dtype=torch.int, device=self.device)
        seq_input[:, 0] = self.decoder.config.bos_token_id
        expanded_return_idx = (
            torch.arange(seq_input.shape[0]).view(-1, 1).repeat(1, beam_num).view(-1).to(self.device)
        )
        encoder_hidden_states = enc_outputs_vision.index_select(0, expanded_return_idx)
        mask_p = torch.ones(batch_size, 1, dtype=torch.int, device=self.device)
        semantics = semantics.index_select(0, expanded_return_idx)
        semantics = semantics.unsqueeze(1)
        model_kwargs = {"encoder_hidden_states": encoder_hidden_states, "inputs_embeds": semantics,"attention_mask": mask_p}

        #generate text
        outputs = self.decoder.generate(input_ids=seq_input,
                                        bos_token_id=self.decoder.config.bos_token_id,
                                        eos_token_id=self.decoder.config.eos_token_id,
                                        pad_token_id=self.decoder.config.pad_token_id,
                                        max_length=self.max_length,
                                        num_beams=beam_num,
                                        num_return_sequences=1, **model_kwargs)

        # always get the output to the same size
        if outputs.size(1) != self.max_length:
            outputs = torch.cat(
                (outputs, torch.zeros(outputs.size(0), self.max_length - outputs.size(1), device=outputs.device)),
                dim=1)
        #compute NLP-Metrics on valdition data
        self.accuracy.update(outputs, dec_inputs)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.log('harmonic_mean_bleu_meteor', self.accuracy.compute(self.val), on_epoch=True, prog_bar=True,logger=True,on_step=False)
        self.log('val/harmonic_mean_bleu_meteor', self.accuracy.harmonice, on_epoch=True, prog_bar=True,logger=True, on_step=False)
        self.log("val/bleu1", self.accuracy.bleu1, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("val/bleu2", self.accuracy.bleu2, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("val/bleu3", self.accuracy.bleu3, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("val/bleu4", self.accuracy.bleu4, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("val/meteor", self.accuracy.meteor, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("val/cider", self.accuracy.cider, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.log("val/rougel", self.accuracy.rougel, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        self.accuracy.reset()

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def setup(self, stage):
        # data
        train_data, val_data, test_data = self.dataset.read_video_with_caption()

        self.train_l = Video_Caption_Loader(dataset=train_data,
                                               path=self.path_data, type=self.dataset.type, vocab=self.vocab,
                                               max_length=self.max_length, config=self.config_data, adaptive=self.using_adaptive)
        self.val_l = Video_Caption_Loader(dataset=val_data,
                                             path=self.path_data, type=self.dataset.type, vocab=self.vocab,
                                             max_length=self.max_length, config=self.config_data, adaptive=self.using_adaptive)
        self.test_l = Video_Caption_Loader(dataset=test_data,
                                             path=self.path_data, type=self.dataset.type, vocab=self.vocab,
                                             max_length=self.max_length, config=self.config_data,
                                             adaptive=self.using_adaptive)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_l, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_l, batch_size=self.batch_size, num_workers=16, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_l, batch_size=self.batch_size, num_workers=16, pin_memory=True)
