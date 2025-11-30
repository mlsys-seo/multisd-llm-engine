from typing import Optional

import torch

from sdprofiler.utils.parallel_group_utils import ParallelGroup

    
    
class TensorParallelEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        weight: torch.Tensor,
        padding_idx: int,
        parallel_group: ParallelGroup
    ):
        num_embeddings, embedding_dim = weight.shape
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, dtype=weight.dtype)
        self.parallel_group = parallel_group
        self.weight.copy_(weight)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output = self.parallel_group.all_gather(output)
        return output



class ColumnParallelLinear(torch.nn.Linear):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        parallel_group: ParallelGroup,
        gather_output: bool = False
    ):
        out_features, in_features = weight.shape
        super().__init__(
            in_features, 
            out_features, 
            bias=True if bias is not None else False, 
            dtype=weight.dtype
        )
        self.parallel_group = parallel_group
        self.gather_output = gather_output

        self.weight.copy_(weight)
        if bias is not None:
            self.bias.copy_(bias)

    def forward(self, input: torch.Tensor):
        output = torch.nn.functional.linear(input, self.weight)
        if self.bias is not None:
            output += self.bias

        if self.gather_output:
            output = self.parallel_group.all_gather(output)

        return output


class RowParallelLinear(torch.nn.Linear):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        parallel_group: ParallelGroup,
    ):
        out_features, in_features = weight.shape
        super().__init__(
            in_features, 
            out_features, 
            bias=True if bias is not None else False, 
            dtype=weight.dtype
        )
        self.parallel_group = parallel_group

        self.weight.copy_(weight)
        if bias is not None:
            self.bias.copy_(bias)

    def forward(self, input: torch.Tensor):
        output = torch.nn.functional.linear(input, self.weight)
        output = self.parallel_group.all_reduce(output)
        if self.bias is not None:
            output += self.bias

        return output



class TensorParallelEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        weight: torch.Tensor,
        padding_idx: int,
        parallel_group: ParallelGroup
    ):
        num_embeddings, embedding_dim = weight.shape
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, dtype=weight.dtype)
        self.parallel_group = parallel_group
        self.weight.copy_(weight)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output = self.parallel_group.all_gather(output)
        return output



class TensorParallelLearnedPositionalEmbedding(torch.nn.Embedding):
    def __init__(
        self, 
        weight: torch.Tensor,
        offset: int,
        parallel_group: ParallelGroup
    ):
        num_embeddings, embedding_dim = weight.shape
        super().__init__(num_embeddings, embedding_dim, dtype=weight.dtype)
        self.parallel_group = parallel_group
        self.offset = offset
        self.weight.copy_(weight)

    def forward(
        self,
        attention_mask: torch.LongTensor,
        past_key_values_length: int = 0,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        if position_ids is None:
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            # cut positions if `past_key_values_length` is > 0
            position_ids = position_ids[:, past_key_values_length:]

        output = super().forward(position_ids + self.offset)
        output = self.parallel_group.all_gather(output)
        return output