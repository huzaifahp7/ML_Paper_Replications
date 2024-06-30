import math
#Type hints
from typing import Optional, List
import torch
#Base class for all neural networks 
from torch import nn
#Importing tracker from labml for debugging
from labml import tracker
import matplotlib.pyplot as plt

class Prepare(nn.Module):
    """ Prepare the input for multi headed attention by doing a linear transformation and splitting the vector into given number of heads 
    for multi headed attention. Used to transfrom query(q), key(k) and value(v) vectors into multiple heads. 
    """
    
    def __init__(self, d_model: int, n_heads: int, d_k: int, bias: bool):
        super().__init__()
        #Linear layer for linear transformation of input vector
        self.linear = nn.Linear(d_model, n_heads * d_k, bias=bias)
        #Number of heads
        self.heads = n_heads
        #Number of dimensions for each head
        self.d_k = d_k
        
    def foward(self,x: torch.Tensor):
        """ Forward pass for Prepare"""
        #Linear transformation to the last dimension of the input vector
        head_shape  = x.shape[:-1] 
        #Linear transformation
        x = self.linear(x)
        #Split the last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True): 
        
        super().__init__()
        #Number of heads
        self.heads = heads
        #Number of features for each head
        self.d_k = d_model // heads
        #Transform query, key, value vectors into multiple heads
        self.query = Prepare(d_model, heads, self.d_k, bias)
        self.key = Prepare(d_model, heads, self.d_k, bias)
        self.value = Prepare(d_model, heads, self.d_k, bias=True)
        
        #Softmax for attention along the time dimension for key
        self.softmax = nn.Softmax(dim=-1)
        #Output Layer
        self.output = nn.Linear(d_model, d_model)
        #Dropout Layer
        self.dropout = nn.Dropout(dropout_prob)
        #Scaling factor before softmax
        self.scale = 1 / math.sqrt(self.d_k)
        #Store attentions so thta it can be used for logging 
        self.attn = None
        
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """ Calculate scores between queries and keys"""
        #Get the dot product of query and key
        return torch.einsum('ibhd,jbhd->ijbh', query, key)
    
    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """ Prepare mask for attention"""
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        
        #Same mask applied to all heads
        mask = mask.unsqueeze(-1)
        #Mask has the shape of [seq_len_q, seq_len_k, heads]
        return mask
    
    def foward(self, *, 
               query: torch.Tensor,
               key: torch.Tensor,
               value: torch.Tensor,
               mask: Optional[torch.Tensor] = None):
        
        """ Forward pass for MultiHeadAttention
        Args:
            query: Query tensor of shape [seq_len, batch_size, d_model]
            key: Key tensor of shape [seq_len, batch_size, d_model]
            value: Value tensor of shape [seq_len, batch_size, d_model]
            mask: Mask tensor of shape [seq_len, seq_len, batch_size]
        """
        
        seq_len, batch_size = query.shape
        
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        
        #Prepare query, key, value for attention computation. These will then have shape [seq_len, batch_size, heads, d_k]
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        #Compute scores
        scores = self.get_scores(query, key)
        #Scale scores
        scores = scores * self.scale
        
        #Apply masking
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        
        #Softmax attention along the key sequence dimension
        attn = self.softmax(scores)
        #Use labml's tracker to save attentions
        tracker.debug('attn', attn)
        
        #Apply dropout
        attn = self.dropout(attn)
        
        #Multiply by values
        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)
        self.attn = attn.detach()
        #Cocatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)
        #Output layer
        return self.output(x)
    
    
# # Create sample input tensors for query, key, and value
# seq_len = 5
# batch_size = 2
# d_model = 8
# heads = 2

# query = torch.rand((seq_len, batch_size, d_model))
# key = torch.rand((seq_len, batch_size, d_model))
# value = torch.rand((seq_len, batch_size, d_model))
# mask = torch.ones((seq_len, seq_len, batch_size))

# # Create an instance of the MultiHeadAttention class
# mha = MultiHeadAttention(heads=heads, d_model=d_model)

# # Pass the sample input through the MultiHeadAttention module
# output = mha(query=query, key=key, value=value, mask=mask)

# # Ensure correct output shape
# assert output.shape == (seq_len, batch_size, d_model), f"Expected shape {(seq_len, batch_size, d_model)}, but got {output.shape}"
# print("Output shape is correct.")

# # Retrieve the attention weights
# attention_weights = mha.attn

# # Visualize the attention weights
# for head in range(heads):
#     plt.matshow(attention_weights[:, :, head].cpu().detach().numpy(), cmap='viridis')
#     plt.title(f'Attention Weights for Head {head+1}')
#     plt.colorbar()
#     plt.show()