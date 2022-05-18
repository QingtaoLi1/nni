# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import overload

import pytorch_lightning as pl
from torch import Tensor


class CompressionLightningModule(pl.LightningModule):
    pass

# class CompressionLightningModule(pl.LightningModule):
#     def training_step(self, batch, batch_idx):
#         return self.compression_training_step(batch, batch_idx, self.criterion)

#     def compression_training_step(self, batch, batch_idx, criterion):
#         """
#         The only different between this function and pytorch_lightning.LightningModule.training_step is
#         this function using a 
#         """
#         # TODO: using logger
#         print('[INFO] CompressionLightningModule is using default compression_training_step()')
#         data, target = batch
#         output = self(data)
#         loss = criterion(output, target)
#         return loss

#     @overload
#     def criterion(self, output, target) -> Tensor:
#         ...

#     def criterion(self, *args, **kwargs) -> Tensor:
#         """
#         Return the loss of the model output and target. Some pruner will add additional loss by patch this function.

#         Example::

#             import torch.nn.functional as F

#             class Net(CompressionLightningModule):
#                 ...

#                 def criterion(self, output, target) -> Tensor:
#                     return F.cross_entropy(output, target)
#         """
#         raise NotImplementedError('`criterion` must be implemented to be used with the NNI Compressor')
