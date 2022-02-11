# Cell
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim

from ..components.embed import DataEmbedding, DataEmbedding_wo_pos
from ..components.autocorrelation import (
    AutoCorrelation, AutoCorrelationLayer
)
from ..components.autoformer import (
    Encoder, Decoder, EncoderLayer, DecoderLayer,
    my_Layernorm, series_decomp
)
from ...losses.utils import LossFunction

# Cell
class _Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, seq_len,
                 label_len, pred_len, output_attention,
                 enc_in, dec_in, d_model, c_out, embed, freq, dropout,
                 factor, n_heads, d_ff, moving_avg, activation, e_layers,
                 d_layers):
        super(_Autoformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq,
                                                  dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq,
                                                  dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

# Cell
class Autoformer(pl.LightningModule):
    def __init__(self, seq_len,
                 label_len, pred_len, output_attention,
                 enc_in, dec_in, d_model, c_out, embed, freq, dropout,
                 factor, n_heads, d_ff, moving_avg, activation, e_layers, d_layers,
                 loss_train, loss_valid, loss_hypar, learning_rate,
                 lr_decay, weight_decay, lr_decay_step_size,
                 random_seed):
        super(Autoformer, self).__init__()

        #------------------------ Model Attributes ------------------------#
        # Architecture parameters
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.d_model = d_model
        self.c_out = c_out
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.factor = factor
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.activation = activation
        self.e_layers = e_layers
        self.d_layers = d_layers

        # Loss functions
        self.loss_train = loss_train
        self.loss_hypar = loss_hypar
        self.loss_valid = loss_valid
        self.loss_fn_train = LossFunction(loss_train,
                                          seasonality=self.loss_hypar)
        self.loss_fn_valid = LossFunction(loss_valid,
                                          seasonality=self.loss_hypar)

        # Regularization and optimization parameters
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.lr_decay_step_size = lr_decay_step_size
        self.random_seed = random_seed

        self.model = _Autoformer(seq_len,
                                 label_len, pred_len, output_attention,
                                 enc_in, dec_in, d_model, c_out,
                                 embed, freq, dropout,
                                 factor, n_heads, d_ff,
                                 moving_avg, activation, e_layers,
                                 d_layers)

    def forward(self, batch):
        """
        Autoformer needs batch of shape (batch_size, time, series) for y
        and (batch_size, time, exogenous) for x
        and doesnt need X for each time series.
        USE DataLoader from pytorch instead of TimeSeriesLoader.
        """
        Y = batch['Y'].permute(0, 2, 1)
        X = batch['X'][:, 0, :, :].permute(0, 2, 1)
        sample_mask = batch['sample_mask'].permute(0, 2, 1)
        available_mask = batch['available_mask']

        s_begin = 0
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        batch_x = Y[:, s_begin:s_end, :]
        batch_y = Y[:, r_begin:r_end, :]
        batch_x_mark = X[:, s_begin:s_end, :]
        batch_y_mark = X[:, r_begin:r_end, :]
        outsample_mask = sample_mask[:, r_begin:r_end, :]

        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1)

        if self.output_attention:
            forecast = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            forecast = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        batch_y = batch_y[:, -self.pred_len:, :]
        outsample_mask = outsample_mask[:, -self.pred_len:, :]

        return batch_y, forecast, outsample_mask

    def training_step(self, batch, batch_idx):

        outsample_y, forecast, outsample_mask = self(batch)

        loss = self.loss_fn_train(y=outsample_y,
                                  y_hat=forecast,
                                  mask=outsample_mask,
                                  y_insample= batch['Y'].permute(0, 2, 1))

        self.log('train_loss', loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, idx):

        outsample_y, forecast, outsample_mask = self(batch)

        loss = self.loss_fn_valid(y=outsample_y,
                                  y_hat=forecast,
                                  mask=outsample_mask,
                                  y_insample= batch['Y'].permute(0, 2, 1))

        self.log('val_loss', loss, prog_bar=True)

        return loss

    def on_fit_start(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=self.lr_decay_step_size,
                                                 gamma=self.lr_decay)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}