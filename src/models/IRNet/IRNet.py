import math
import numpy as np

import torch as t
import torch.nn as nn

from typing import Tuple
from functools import partial

from ..components.tcn import _TemporalConvNet
from ..components.common import Chomp1d, RepeatVector
from ...losses.utils import LossFunction

# Cell
class _StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(_StaticFeaturesEncoder, self).__init__()
        layers = [nn.Dropout(p=0.5),
                  nn.Linear(in_features=in_features, out_features=out_features),
                  nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x

class _sEncoder(nn.Module):
    def __init__(self, in_features, out_features, n_time_in):
        super(_sEncoder, self).__init__()
        layers = [nn.Dropout(p=0.5),
                  nn.Linear(in_features=in_features, out_features=out_features),
                  nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.repeat = RepeatVector(repeats=n_time_in)

    def forward(self, x):
        # Encode and repeat values to match time
        x = self.encoder(x)
        x = self.repeat(x) # [N,S_out] -> [N,S_out,T]
        return x

# Cell
class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, -self.forecast_size:]
        return backcast, forecast
    
    
# Cell
def init_weights(module, initialization):
    if type(module) == t.nn.Linear:
        if initialization == 'orthogonal':
            t.nn.init.orthogonal_(module.weight)
        elif initialization == 'he_uniform':
            t.nn.init.kaiming_uniform_(module.weight)
        elif initialization == 'he_normal':
            t.nn.init.kaiming_normal_(module.weight)
        elif initialization == 'glorot_uniform':
            t.nn.init.xavier_uniform_(module.weight)
        elif initialization == 'glorot_normal':
            t.nn.init.xavier_normal_(module.weight)
        elif initialization == 'lecun_normal':
            pass #t.nn.init.normal_(module.weight, 0.0, std=1/np.sqrt(module.weight.numel()))
        else:
            assert 1<0, f'Initialization {initialization} not found'

# Cell
ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']


class _TreeDRNetBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self, n_time_in: int, n_time_out: int, n_x: int,
                 n_s: int, n_s_hidden: int, n_theta: int, n_theta_hidden: list,
                 basis: nn.Module,
                 n_layers: int,  batch_normalization: bool, dropout_prob: float, activation: str):
        """
        """
        super().__init__()

        if n_s == 0:
            n_s_hidden = 0
        n_theta_hidden = [n_time_in + (n_time_in+n_time_out)*n_x + n_s_hidden] + n_theta_hidden

        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.n_s = n_s
        self.n_s_hidden = n_s_hidden
        self.n_x = n_x
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        
        self.treedrnet = NBeats_boosting_trees_1(
                 forecast_length = self.n_time_out,
                 backcast_length=self.n_time_in,
                 out_size=self.n_s_hidden,
                 inner_size=128,
                 stacks= 2,
                 blocks_per_stack=2,
                 duplicate = 2,
                 depth = 2,
                 outer_loop = 2,
                 boosting_round = 1,
                 dropout_prob = self.dropout_prob)

        
        
    def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor,
        outsample_x_t: t.Tensor, x_s: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:

        batch_size = len(insample_y)
        if self.n_x > 0:
            insample_y = t.cat(( insample_y, insample_x_t.reshape(batch_size, -1) ), 1)
            insample_y = t.cat(( insample_y, outsample_x_t.reshape(batch_size, -1) ), 1)

        # Static exogenous
        if (self.n_s > 0) and (self.n_s_hidden > 0):
            x_s = self.static_encoder(x_s)
            insample_y = t.cat((insample_y, x_s), 1)
        #print("^^^^^^^^^^^^",batch_size,insample_y.shape)
        backcast, forecast = self.treedrnet(insample_y)
        return backcast, forecast
    
    
# Cell
class _TreeDRNet(nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self,
                 n_time_in,
                 n_time_out,
                 n_s,
                 n_x,
                 n_s_hidden,
                 n_x_hidden,
                 n_polynomials,
                 n_harmonics,
                 stack_types: list,
                 n_blocks: list,
                 n_layers: list,
                 n_theta_hidden: list,
                 dropout_prob_theta,
                 activation,
                 initialization,
                 batch_normalization,
                 shared_weights):
        super().__init__()

        self.n_time_out = n_time_out

        self.blocks = self.create_stack(stack_types=stack_types,
                                   n_blocks=n_blocks,
                                   n_time_in=n_time_in,
                                   n_time_out=n_time_out,
                                   n_x=n_x,
                                   n_x_hidden=n_x_hidden,
                                   n_s=n_s,
                                   n_s_hidden=n_s_hidden,
                                   n_layers=n_layers,
                                   n_theta_hidden=n_theta_hidden,
                                   batch_normalization=batch_normalization,
                                   dropout_prob_theta=dropout_prob_theta,
                                   activation=activation,
                                   shared_weights=shared_weights,
                                   n_polynomials=n_polynomials,
                                   n_harmonics=n_harmonics,
                                   initialization=initialization)
        #self.blocks = t.nn.ModuleList(blocks)

    def create_stack(self, stack_types, n_blocks,
                     n_time_in, n_time_out,
                     n_x, n_x_hidden, n_s, n_s_hidden,
                     n_layers, n_theta_hidden, batch_normalization, dropout_prob_theta,
                     activation, shared_weights,
                     n_polynomials, n_harmonics, initialization):

        nbeats_block = _TreeDRNetBlock(n_time_in=n_time_in,
                                    n_time_out=n_time_out,
                                    n_x=n_x,
                                    n_s=n_s,
                                    n_s_hidden=n_s_hidden,
                                    n_theta=None,
                                    n_theta_hidden=n_theta_hidden,
                                    basis=None,
                                    n_layers=n_layers,
                                    batch_normalization=None,
                                    dropout_prob=dropout_prob_theta,
                                    activation=activation)

                # Select type of evaluation and apply it to all layers of block
        init_function = partial(init_weights, initialization=initialization)
        nbeats_block.apply(init_function)

        return nbeats_block

    def forward(self, S: t.Tensor, Y: t.Tensor, X: t.Tensor,
                insample_mask: t.Tensor, outsample_mask: t.Tensor,
                return_decomposition: bool=False):

        # insample
        insample_y    = Y[:, :-self.n_time_out]
        insample_x_t  = X[:, :, :-self.n_time_out]
        insample_mask = insample_mask[:, :-self.n_time_out]

        # outsample
        outsample_y   = Y[:, -self.n_time_out:]
        outsample_x_t = X[:, :, -self.n_time_out:]
        outsample_mask = outsample_mask[:, -self.n_time_out:]

        if return_decomposition:
            forecast, block_forecasts = self.forecast_decomposition(insample_y=insample_y,
                                                                    insample_x_t=insample_x_t,
                                                                    insample_mask=insample_mask,
                                                                    outsample_x_t=outsample_x_t,
                                                                    x_s=S)
            return outsample_y, forecast, block_forecasts, outsample_mask

        else:
            forecast = self.forecast(insample_y=insample_y,
                                     insample_x_t=insample_x_t,
                                     insample_mask=insample_mask,
                                     outsample_x_t=outsample_x_t,
                                     x_s=S)
            return outsample_y, forecast, outsample_mask

    def forecast(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
                 outsample_x_t: t.Tensor, x_s: t.Tensor):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:] # Level with Naive1
        
        # for i, block in enumerate(self.blocks):
        #     backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
        #                                      outsample_x_t=outsample_x_t, x_s=x_s)
        #     residuals = (residuals - backcast) * insample_mask
        #     forecast = forecast + block_forecast
        backcast, forecast = self.blocks(insample_y=residuals, insample_x_t=insample_x_t,
                                              outsample_x_t=outsample_x_t, x_s=x_s)

        return forecast

    def forecast_decomposition(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
                               outsample_x_t: t.Tensor, x_s: t.Tensor):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        n_batch, n_channels, n_t = outsample_x_t.size(0), outsample_x_t.size(1), outsample_x_t.size(2)

        level = insample_y[:, -1:] # Level with Naive1
        block_forecasts = [ level.repeat(1, n_t) ]

        forecast = level
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
                                             outsample_x_t=outsample_x_t, x_s=x_s)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        # (n_batch, n_blocks, n_t)
        block_forecasts = t.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1,0,2)

        return forecast, block_forecasts
    
    

    
# Cell
import time
import random
from copy import deepcopy
from collections import defaultdict

from torch import optim
import pytorch_lightning as pl

from ...losses.pytorch import (
    MAPELoss, MASELoss, SMAPELoss,
    MSELoss, MAELoss, PinballLoss
)
from ...losses.numpy import (
    mae, mse, mape,
    smape, rmse, pinball_loss
)

from ...data.tsdataset import WindowsDataset

# Cell
class IRNet(pl.LightningModule):
    def __init__(self,
                 n_time_in,
                 n_time_out,
                 n_x,
                 n_x_hidden,
                 n_s,
                 n_s_hidden,
                 shared_weights,
                 activation,
                 initialization,
                 stack_types,
                 n_blocks,
                 n_layers,
                 n_harmonics,
                 n_polynomials,
                 n_theta_hidden,
                 batch_normalization,
                 dropout_prob_theta,
                 learning_rate,
                 lr_decay,
                 lr_decay_step_size,
                 weight_decay,
                 loss_train,
                 loss_hypar,
                 loss_valid,
                 frequency,
                 random_seed,
                 seasonality):
        super(IRNet, self).__init__()
        """
        N-BEATS model.

        Parameters
        ----------
        # TODO: Fix parameters' documentation.
        # TODO: Remove useless parameters (dropout_prob_exogenous).
        n_time_in: int
            Multiplier to get insample size.
            Insample size = n_time_in * output_size
        n_time_out: int
            Forecast horizon.
        shared_weights: bool
            If True, repeats first block.
        activation: str
            Activation function.
            An item from ['relu', 'softplus', 'tanh', 'selu', 'lrelu', 'prelu', 'sigmoid'].
        initialization: str
            Initialization function.
            An item from ['orthogonal', 'he_uniform', 'glorot_uniform', 'glorot_normal', 'lecun_normal'].
        stack_types: List[str]
            List of stack types.
            Subset from ['seasonality', 'trend', 'identity', 'exogenous', 'exogenous_tcn', 'exogenous_wavenet'].
        n_blocks: List[int]
            Number of blocks for each stack type.
            Note that len(n_blocks) = len(stack_types).
        n_layers: List[int]
            Number of layers for each stack type.
            Note that len(n_layers) = len(stack_types).
        n_hidden: List[List[int]]
            Structure of hidden layers for each stack type.
            Each internal list should contain the number of units of each hidden layer.
            Note that len(n_hidden) = len(stack_types).
        n_harmonics: List[int]
            Number of harmonic terms for each stack type.
            Note that len(n_harmonics) = len(stack_types).
        n_polynomials: List[int]
            Number of polynomial terms for each stack type.
            Note that len(n_polynomials) = len(stack_types).
        exogenous_n_channels:
            Exogenous channels for non-interpretable exogenous basis.
        batch_normalization: bool
            Whether perform batch normalization.
        dropout_prob_theta: float
            Float between (0, 1).
            Dropout for Nbeats basis.
        dropout_prob_exogenous: float
            Float between (0, 1).
            Dropout for exogenous basis.
        x_s_n_hidden: int
            Number of encoded static features to calculate.
        learning_rate: float
            Learning rate between (0, 1).
        lr_decay: float
            Decreasing multiplier for the learning rate.
        lr_decay_step_size: int
            Steps between each lerning rate decay.
        weight_decay: float
            L2 penalty for optimizer.
        loss_train: str
            Loss to optimize.
            An item from ['MAPE', 'MASE', 'SMAPE', 'MSE', 'MAE', 'PINBALL', 'PINBALL2'].
        loss_hypar:
            Hyperparameter for chosen loss.
        loss_valid:
            Validation loss.
            An item from ['MAPE', 'MASE', 'SMAPE', 'RMSE', 'MAE', 'PINBALL'].
        frequency: str
            Time series frequency.
        random_seed: int
            random_seed for pseudo random pytorch initializer and
            numpy random generator.
        seasonality: int
            Time series seasonality.
            Usually 7 for daily data, 12 for monthly data and 4 for weekly data.
        """

        if activation == 'SELU': initialization = 'lecun_normal'

        #------------------------ Model Attributes ------------------------#
        # Architecture parameters
        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.n_x = n_x
        self.n_x_hidden = n_x_hidden
        self.n_s = n_s
        self.n_s_hidden = n_s_hidden
        self.shared_weights = shared_weights
        self.activation = activation
        self.initialization = initialization
        self.stack_types = stack_types
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_harmonics = n_harmonics
        self.n_polynomials = n_polynomials
        self.n_theta_hidden = n_theta_hidden

        # Loss functions
        self.loss_train = loss_train
        self.loss_hypar = loss_hypar
        self.loss_valid = loss_valid
        self.loss_fn_train = LossFunction(loss_train,
                                          seasonality=self.loss_hypar)
        self.loss_fn_valid = LossFunction(loss_valid,
                                          seasonality=self.loss_hypar)

        # Regularization and optimization parameters
        self.batch_normalization = batch_normalization
        self.dropout_prob_theta = dropout_prob_theta
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.lr_decay_step_size = lr_decay_step_size
        self.random_seed = random_seed

        # Data parameters
        self.frequency = frequency
        self.seasonality = seasonality
        self.return_decomposition = False

        self.model = _TreeDRNet(n_time_in=self.n_time_in,
                             n_time_out=self.n_time_out,
                             n_s=self.n_s,
                             n_x=self.n_x,
                             n_s_hidden=self.n_s_hidden,
                             n_x_hidden=self.n_x_hidden,
                             n_polynomials=self.n_polynomials,
                             n_harmonics=self.n_harmonics,
                             stack_types=self.stack_types,
                             n_blocks=self.n_blocks,
                             n_layers=self.n_layers,
                             n_theta_hidden=self.n_theta_hidden,
                             dropout_prob_theta=self.dropout_prob_theta,
                             activation=self.activation,
                             initialization=self.initialization,
                             batch_normalization=self.batch_normalization,
                             shared_weights=self.shared_weights)
        

    def training_step(self, batch, batch_idx):
        S = batch['S']
        Y = batch['Y']
        X = batch['X']
        sample_mask = batch['sample_mask']
        available_mask = batch['available_mask']
        # print('batch size',Y.shape)
        # raise Exception('aaa')

        outsample_y, forecast, outsample_mask = self.model(S=S, Y=Y, X=X,
                                                           insample_mask=available_mask,
                                                           outsample_mask=sample_mask,
                                                           return_decomposition=False)

        loss = self.loss_fn_train(y=outsample_y,
                                  y_hat=forecast,
                                  mask=outsample_mask,
                                  y_insample=Y)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        
        # mem_params = sum([param.nelement()*param.element_size() for param in self.model.parameters()])
        # mem_bufs = sum([buf.nelement()*buf.element_size() for buf in self.model.buffers()])
        # mem = mem_params + mem_bufs
        # print('parameter usuage',mem)
        # print('mem usuage',t.cuda.max_memory_allocated()/1024**2)
        # raise Exception('aaa')

        return loss

    def validation_step(self, batch, idx):
        S = batch['S']
        Y = batch['Y']
        X = batch['X']
        sample_mask = batch['sample_mask']
        available_mask = batch['available_mask']

        outsample_y, forecast, outsample_mask = self.model(S=S, Y=Y, X=X,
                                                           insample_mask=available_mask,
                                                           outsample_mask=sample_mask,
                                                           return_decomposition=False)
        
        

        loss = self.loss_fn_valid(y=outsample_y,
                                  y_hat=forecast,
                                  mask=outsample_mask,
                                  y_insample=Y)

        self.log('val_loss', loss, prog_bar=True)

        return loss

    def on_fit_start(self):
        t.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed) #TODO: interaccion rara con window_sampling de validacion

    def forward(self, batch):
        S = batch['S']
        Y = batch['Y']
        X = batch['X']
        sample_mask = batch['sample_mask']
        available_mask = batch['available_mask']

        if self.return_decomposition:
            outsample_y, forecast, block_forecast, outsample_mask = self.model(S=S, Y=Y, X=X,
                                                                     insample_mask=available_mask,
                                                                     outsample_mask=sample_mask,
                                                                     return_decomposition=True)
            return outsample_y, forecast, block_forecast, outsample_mask

        outsample_y, forecast, outsample_mask = self.model(S=S, Y=Y, X=X,
                                                           insample_mask=available_mask,
                                                           outsample_mask=sample_mask,
                                                           return_decomposition=False)
        return outsample_y, forecast, outsample_mask

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=self.lr_decay_step_size,
                                                 gamma=self.lr_decay)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    
    
    
    
    
    
class NBeats_boosting_trees_1(nn.Module):
    def __init__(self,
                forecast_length=5,
                 backcast_length=10,
                 out_size=8,
                 inner_size=256,
                 stacks= 3,
                 blocks_per_stack=3,
                 duplicate = 1,
                 depth = 2,
                 outer_loop = 2,
                 boosting_round = 2,
                 dropout_prob = 0):
            
        super(NBeats_boosting_trees_1, self).__init__()

        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.inner_size = inner_size
        self.out_size = out_size



        self.stack_number = stacks
        self.blocks_per_stack = blocks_per_stack


        self.duplicate = duplicate
        self.depth = depth
        
        self.outer_loop = outer_loop
        self.boosting_round = boosting_round
        self.dropout_prob = dropout_prob
        
        #         self.theta_b_fc = nn.Linear(self.inner_size ,  self.backcast_length )
        #         self.theta_f_fc = nn.Linear(self.inner_size , self.out_size)
        self.stacks = nn.ModuleList()
        self.reuse_forecast_stacks = nn.ModuleList()
        
        
        
#         stacks = []      
        for i in range(self.boosting_round):
            self.stacks.append(Parallel_multi_Tree_controled_NBeats(
                                        forecast_length= self.forecast_length,
                                        backcast_length=self.backcast_length, 
                                        out_size= self.forecast_length,
                                        inner_size= self.inner_size,
                                        stacks= self.stack_number,
                                        blocks_per_stack= self.blocks_per_stack,
                                        duplicate =  self.duplicate,
                                        depth = self.depth,
                                        outer_loop = self.outer_loop,
                                        dropout_prob = self.dropout_prob
                                        )
                              )
            self.reuse_forecast_stacks.append(self.basic_block_build())
            
        self.input_tranform()
    
#         self.stacks = nn.Sequential(*stacks)
    def input_tranform(self):
        self.layer_1 = nn.Linear(self.backcast_length, 512)
        self.layer_2 = nn.Linear(512,self.inner_size)
            
    def basic_block_build(self):
        stacks = []
        for i in range(self.depth):
            if i == 0:
                stacks.append(nn.Linear(self.forecast_length,self.inner_size))
            else:
                stacks.append(nn.Linear(self.inner_size,self.inner_size))
            stacks.append(nn.ReLU())
        stacks.append(nn.Linear(self.inner_size, self.backcast_length))
        return nn.Sequential(*stacks)
        
    def forward(self, x):

#         *pos_idxs
#         x =  self.batchnorm(x)
#         x = self.layer_2(self.layer_1(x))
       
        for i in range(self.boosting_round):             
            x_tmp,y_tmp = self.stacks[i](x)
            if i == 0:
                y = y_tmp
            else:
                y = y+y_tmp
            x = x-x_tmp - self.reuse_forecast_stacks[i](y)
#             x = x - x_tmp
                
#             backcast = backcast - self.theta_b_fc(x)
                
#             if i == 0:
#                 y =  y_tmp#self.theta_f_fc(x)
#             else:
#                 y = y + y_tmp#self.theta_f_fc(x)#*0.9**(i)
        
#         y =t.mean(t.stack(y_tmp_list_last,dim = 2),dim=2)
#         print(y.shape,y_tmp_left.shape)
    
        return x,y


class NBeats_boosting_trees(nn.Module):
    def __init__(self,
                 forecast_length=5,
                 backcast_length=10,
                 out_size=8,
                 inner_size=256,
                 stacks= 3,
                 blocks_per_stack=3,
                 duplicate = 1,
                 depth = 4,
                 outer_loop = 2,
                 boosting_round = 2,
                 dropout_prob = 0):
            
        super(NBeats_boosting_trees, self).__init__()

        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.inner_size = inner_size
        self.out_size = out_size



        self.stack_number = stacks
        self.blocks_per_stack = blocks_per_stack


        self.duplicate = duplicate
        self.depth = depth

        self.outer_loop = outer_loop
        self.boosting_round = boosting_round
        self.dropout_prob = dropout_prob
        #         self.theta_b_fc = nn.Linear(self.inner_size ,  self.backcast_length )
        #         self.theta_f_fc = nn.Linear(self.inner_size , self.out_size)
        self.stacks = nn.ModuleList()
#         stacks = []      
        for i in range(self.boosting_round):
            self.stacks.append(Parallel_multi_Tree_controled_NBeats(
                                        forecast_length= self.forecast_length,
                                        backcast_length=self.backcast_length, 
                                        out_size= self.forecast_length,
                                        inner_size= self.inner_size,
                                        stacks= self.stack_number,
                                        blocks_per_stack= self.blocks_per_stack,
                                        duplicate =  self.duplicate,
                                        depth = self.depth,
                                        outer_loop = self.outer_loop,
                                        dropout_prob = self.dropout_prob
                                        )
                              )
    
#         self.stacks = nn.Sequential(*stacks)
        self.batchnorm=nn.BatchNorm1d(self.forecast_length)
            
  
        
    def forward(self, x):

#         *pos_idxs
#         x =  self.batchnorm(x)
        x = self.batchnorm(x)
        
       
        for i in range(self.boosting_round):             
            x_tmp,y_tmp = self.stacks[i](x)
            if i == 0:
                x = x-x_tmp
                y = y_tmp
            else:
                x = x-x_tmp
                y = y+y_tmp
#             x = x - x_tmp
                
#             backcast = backcast - self.theta_b_fc(x)
                
#             if i == 0:
#                 y =  y_tmp#self.theta_f_fc(x)
#             else:
#                 y = y + y_tmp#self.theta_f_fc(x)#*0.9**(i)
        
#         y =t.mean(t.stack(y_tmp_list_last,dim = 2),dim=2)
#         print(y.shape,y_tmp_left.shape)
    
        return x,y
    

    

    
    
class Parallel_multi_Tree_controled_NBeats(nn.Module):
    def __init__(self,
                 forecast_length=5,
                 backcast_length=10,
                 out_size=8,
                 inner_size=256,
                 stacks= 3,
                 blocks_per_stack=3,
                 duplicate = 1,
                 depth = 4,
                 outer_loop = 2,
                 dropout_prob = 0):
            
        super(Parallel_multi_Tree_controled_NBeats, self).__init__()

        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.inner_size = inner_size
        self.out_size = out_size



        self.stack_number = stacks
        self.blocks_per_stack = blocks_per_stack


        self.duplicate = duplicate
        self.depth = depth

        self.outer_loop = outer_loop
        #         self.theta_b_fc = nn.Linear(self.inner_size ,  self.backcast_length )
        #         self.theta_f_fc = nn.Linear(self.inner_size , self.out_size)
        self.stacks = nn.ModuleList()
        self.dropout_prob = dropout_prob
        

        for i in range(2**self.outer_loop-1):
            self.stacks.append(Parallel_Tri_controled_NBeats_new(
                                        forecast_length= self.forecast_length,
                                        backcast_length=self.backcast_length, 
                                        out_size= self.forecast_length,
                                        inner_size= self.inner_size,
                                        stacks= self.stack_number,
                                        blocks_per_stack= self.blocks_per_stack,
                                        duplicate =  self.duplicate,
                                        depth = self.depth,
                                        dropout_prob = self.dropout_prob
                                        )
                              )

        
        
    
               
    def forward(self, x):
        x_tmp,y_tmp_left,x_tmp_list_left,y_tmp_list_left = self.stacks[0](x)
        x_tmp_list_last = [x_tmp]
#         if self.outer_loop >1:
        y_tmp_list_last = [y_tmp_left]
#         else:
#             y_tmp_list_last = []
        for i in range(1,self.outer_loop):
            start_point = int(2**(i-1))
            end_point = int(2**i)
            for j in range(start_point,end_point):               
                x_tmp,y_tmp_left,x_tmp_list_left,y_tmp_list_left = self.stacks[(j-start_point)*2+end_point-1](x_tmp_list_last[j-1])
                x_tmp,y_tmp_right,x_tmp_list_right,y_tmp_list_right = self.stacks[(j-start_point)*2+end_point](x_tmp_list_last[j-1])
                x_tmp_list_last.append(x_tmp_list_left[:,:,0])
                x_tmp_list_last.append(x_tmp_list_left[:,:,1])
                x_tmp_list_last.append(x_tmp_list_right[:,:,0])
                x_tmp_list_last.append(x_tmp_list_right[:,:,1])
#                 if i == self.outer_loop-1:
                y_tmp_list_last.append(y_tmp_left)
                y_tmp_list_last.append(y_tmp_right)
#             x = x - x_tmp
                
#             backcast = backcast - self.theta_b_fc(x)
                
#             if i == 0:
#                 y =  y_tmp#self.theta_f_fc(x)
#             else:
#                 y = y + y_tmp#self.theta_f_fc(x)#*0.9**(i)
        
        y =t.mean(t.stack(y_tmp_list_last,dim = 2),dim=2)
#         print(y.shape,y_tmp_left.shape)
        x =t.mean(t.stack(x_tmp_list_last,dim = 2),dim=2)
        return x,y



class Parallel_Tri_controled_NBeats_new(nn.Module):
    def __init__(self,
                 forecast_length=5,
                 backcast_length=10,
                 out_size=8,
                 inner_size=256,
                 stacks= 3,
                 blocks_per_stack=3,
                 duplicate = 1,
                 depth = 4,
                 dropout_prob = 0):
            
        super(Parallel_Tri_controled_NBeats_new, self).__init__()

        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        
        
        self.inner_size = inner_size
        self.out_size = out_size



        self.stack_number = stacks
        self.blocks_per_stack = blocks_per_stack


        self.duplicate = duplicate
        self.depth = depth
        self.dropout_prob = dropout_prob

        #         self.theta_b_fc = nn.Linear(self.inner_size ,  self.backcast_length )
        #         self.theta_f_fc = nn.Linear(self.inner_size , self.out_size)
        self.stacks =Parallel_NBeatsNet(stacks= self.stack_number ,
                                        blocks_per_stack= self.blocks_per_stack,
                                        forecast_length= self.forecast_length,
                                        backcast_length= self.backcast_length,
                                        out_size= self.forecast_length,
                                        inner_size= self.inner_size,
                                        duplicate = self.duplicate,
                                        depth =self.depth,
                                       dropout_prob = self.dropout_prob)

        self.controler = Parallel_Controler(input_dim =  self.backcast_length,
                                            inner_dim = self.inner_size,
                                            controler_size = self.duplicate,
                                            depth = self.depth,
                                           dropout_prob = self.dropout_prob)
        self.create_pos_embedding()
        self.input_tranform()
        self.backcast_tranform()
        
    def input_tranform(self,low_rankness = 8,device = 'cuda'):
        self.layer_1 = nn.Linear(self.backcast_length, low_rankness)
        self.layer_2 =  nn.Linear(low_rankness,self.inner_size)
        
    def backcast_tranform(self,low_rankness = 8,device = 'cuda'):
        self.layer_3 = nn.Linear(self.inner_size, low_rankness)
        self.layer_4 =  nn.Linear(low_rankness,self.backcast_length)
        
        
    def create_pos_embedding(self,device = 'cuda'):
        self.scale_embedding = nn.Embedding( self.backcast_length, 1)
        self.locate_embedding = nn.Embedding( self.backcast_length, 1)
        self.noise_embedding = nn.Embedding( self.backcast_length, 1)
        self.pos_idx_set = t.from_numpy(np.array([i for i in range( self.backcast_length)])).to(device)
        

    def forward(self, x):
#         print(x.shape)
#         print(self.layer_1)
#         print(self.layer_2)
#         x = self.layer_2(self.layer_1(x))
        
        pos_idxs =  self.pos_idx_set.repeat(x.shape[0],1)
#         print(pos_idxs.shape)
        scaling_embedding = self.scale_embedding(pos_idxs).squeeze(-1)
        locate_embedding = self.locate_embedding(pos_idxs).squeeze(-1)
        noise_embedding =  self.noise_embedding(pos_idxs).squeeze(-1)
#         print(x.shape,locate_embedding.shape,pos_idxs.shape)
#         err = t.randn(size = noise_embedding.shape).to('cuda')
#         x = (x+locate_embedding)*t.exp(scaling_embedding)#+ err*t.exp(noise_embedding)
    
        model_weights =  self.controler(x)    
        backcast = x.repeat(1,self.duplicate).unsqueeze(-1) * (model_weights.view(x.shape[0],-1,1))
        
        backcast_tmp, forecast,backcast_mean, forecast_mean = self.stacks(backcast)
        backcast = backcast - backcast_tmp
#         print(backcast.shape,backcast_mean.mean)
#         print(self.layer_3)
#         print(self.layer_4)
#         backcast = self.layer_4(self.layer_3(backcast))
#         backcast_mean = self.layer_3(self.layer_3(backcast_mean))
        return backcast_mean,forecast_mean, backcast.view(x.shape[0],-1,self.duplicate), forecast.view(x.shape[0],-1,self.duplicate)
    


        
class Parallel_Controler(nn.Module):
    def __init__(self,
                input_dim,
                inner_dim,
                controler_size = 3,
                depth = 4,
                parallel = False,
                dropout_prob = 0):
        super(Parallel_Controler, self).__init__()
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.controler_size = controler_size
        self.depth = depth
        self.parallel = parallel
        self.dropout_prob = dropout_prob
        
        
        self.control_softmax = nn.Softmax(dim = 2)
        
        self.stacks = self.single_router()
        
    def single_router(self):
        router = []#nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
#                 router.append(nn.Linear(self.input_dim, self.inner_dim)) 
                router.append(nn.Conv1d(
                                in_channels=self.input_dim * self.controler_size, 
                                out_channels=self.inner_dim * self.controler_size, 
                                kernel_size=1, 
                                groups=self.controler_size)
                             )
#             elif i == self.depth -1:
            else:
                router.append(nn.Conv1d(
                                in_channels=self.inner_dim * self.controler_size, 
                                out_channels=self.inner_dim * self.controler_size, 
                                kernel_size=1, 
                                groups=self.controler_size)
                )
            router.append(nn.ReLU())
            if self.dropout_prob>0:
                    router.append(nn.Dropout(p=self.dropout_prob))
        router.append(nn.Conv1d(
                                in_channels=self.inner_dim * self.controler_size, 
                                out_channels=self.input_dim * self.controler_size, 
                                kernel_size=1, 
                                groups=self.controler_size)
                             )
        if self.dropout_prob>0:
                    stacks.append(nn.Dropout(p=self.dropout_prob))
        return nn.Sequential(*router)
  
    def forward(self, x_input):
        x = x_input.repeat(1,self.controler_size).unsqueeze(-1)
#         print(x.shape,x_input.shape)
        flat = self.stacks(x)
        batch_size = x.shape[0]
        return self.control_softmax(flat.view(batch_size, -1, self.controler_size)).view(batch_size, -1,self.controler_size)
    
    
class Parallel_NBeatsNet(nn.Module):
    def __init__(self,
                forecast_length=5,
                 backcast_length=10,
                 out_size=8,
                 inner_size=256,
                 stacks= 3,
                 blocks_per_stack=3,
                 depth = 4,
                 duplicate = 1,
                 dropout_prob=0):
        
        super(Parallel_NBeatsNet, self).__init__()
        
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.inner_size = inner_size
        self.out_size = out_size
        
        self.stack_number = stacks
        self.blocks_per_stack = blocks_per_stack
        
        self.depth = depth
        self.duplicate = duplicate
        self.dropout_prob = dropout_prob
        
        
        blocks = []
        for i in range(self.stack_number):
            for block_id in range(self.blocks_per_stack):
                #if block_id == 0:
                block = Parallel_Block(self.inner_size, self.out_size,self.backcast_length,self.duplicate, self.depth,self.dropout_prob)
                blocks.append(block)
#         blocks = nn.Sequential(*blocks)
        self.stacks = nn.Sequential(*blocks)
 

    def forward(self, backcast):  
#         backcast = backcast.repeat(1,self.duplicate).unsqueeze(-1)
#         print('asdfasdfasdfasdf',backcast.shape)
        backcast,forecast = self.stacks(backcast)
        backcast_mean = t.mean(backcast.view(backcast.shape[0], -1, self.duplicate),dim = 2)
        forecast_mean = t.mean(forecast.view(backcast.shape[0], -1, self.duplicate),dim = 2)
        return backcast, forecast,backcast_mean/(self.stack_number*self.blocks_per_stack), forecast_mean/(self.stack_number*self.blocks_per_stack)
        #return backcast, forecast,backcast_mean, forecast_mean
    
            
                
class Parallel_Block(nn.Module):

    def __init__(self, inner_dim, out_dim, input_dim,heads=1,depth = 4,dropout_prob=0):
        super(Parallel_Block, self).__init__()
        self.inner_dim = inner_dim
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.heads = heads
        self.depth = depth
        self.dropout_prob = dropout_prob
        
        self.basic_block = self.basic_block_build()

        self.theta_b_fc =  nn.Conv1d(in_channels=self.inner_dim * self.heads, 
                             out_channels=self.input_dim * self.heads, 
                             kernel_size=1, 
                             groups=self.heads)

        self.theta_f_fc =  nn.Conv1d(in_channels=self.inner_dim * self.heads, 
                             out_channels=self.out_dim * self.heads, 
                             kernel_size=1, 
                             groups=self.heads)

    def basic_block_build(self):
        stacks = []
        for i in range(self.depth):
            if i == 0:
                #print("HHHHHHH",self.input_dim,self.inner_dim,self.heads)
                stacks.append(nn.Conv1d(in_channels=self.input_dim * self.heads, 
                             out_channels=self.inner_dim * self.heads, 
                             kernel_size=1, 
                             groups=self.heads))
            else:
                stacks.append(nn.Conv1d(in_channels=self.inner_dim * self.heads, 
                             out_channels=self.inner_dim * self.heads, 
                             kernel_size=1, 
                             groups=self.heads))

            stacks.append(nn.ReLU())
            if self.dropout_prob>0:
                    stacks.append(nn.Dropout(p=self.dropout_prob))
        return nn.Sequential(*stacks)
  
    def forward(self, x):
        if isinstance(x,tuple):
            y = x[1]
            x = x[0]
            x1 = self.basic_block(x)
            f = self.theta_f_fc(x1)
            b = self.theta_b_fc(x1)
            return x-b,y+f
        else:
            x1 = self.basic_block(x)
            f = self.theta_f_fc(x1)
            b = self.theta_b_fc(x1)
            return x-b,f
