"""
This file contains the implementation of the transformer model. This is fully inspired by the original paper
"Attention is All You Need" by Vaswani et al. (2017). The original paper can be found here: https://arxiv.org/abs/1706.03762
"""



import torch
from torch import Tensor
import torch.nn as nn
from math import sqrt
from typing import Tuple

#Explained in Section 3.4
class InputEmbedding(nn.Module):
    """
    Take a sentence like "I am a student", then this embeds the words into vectors of embedding_dim dimensions.
    Hence, it returns to us for "I" something like (0.213, 0.392138, 0.8312383, ....) so on so forth.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        #Note the embedding layer is just a mapping from each input to a certain number/vector of numbers.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x) * sqrt(self.embedding_dim)

#Explained in Section 3.5
class PositionalEncoding(nn.Module):
    """
    This is the positional encoding module. It adds the positional information to the input embeddings.
    The reason we need this is because the transformer architecture does not have any recurrence or convolution, hence it does not have any notion of order.
    So we need to add the positional information to the input embeddings so that the model can learn the order of the words in the sentence.
    """

    def __init__(self, seq_len: int, dropout: float, embedding_dim: int) -> None: #dropout avoids overfitting, seq_len is the maximum length of the input sequence.
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #Create a matrix of shape (seq_len, embedding_dim) to hold the positional encodings of ALL words
        pe = torch.zeros(seq_len, embedding_dim)

        #Create a vector of shape (embedding_dim,1) to hold the positional encodings for each dimension
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #This is the position of each word in the sentence, it will be used to calculate the positional encoding for each word.
        div_term = torch.pow(10000.0, torch.arange(0, embedding_dim, 2).float()) #This is the denominator term in the positional encoding formula, it will be used to calculate the positional encoding for each dimension.

        #Calculate the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        #Add a batch dimension to the positional encodings
        pe = pe.unsqueeze(0) #Tensor of (1, seq_len, embedding_dim)

        #Register the positional encodings as a buffer so that they are not updated during training
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        #Add the positional encodings to the input embeddings
        x = x + self.state_dict()['pe'][:, :x.size(1)]
        return self.dropout(x)


class LayerNorm(nn.Module):
    """
    This is the layer normalization module. It normalizes the input to have zero mean and unit variance.
    This is done to stabilize the training of the model and to improve the convergence.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1)) #Scale parameter
        self.bias = nn.Parameter(torch.zeros(1)) #Shift parameter

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

#Explained in Section 3.3
class FeedForward(nn.Module):
    """
    This is the feed forward module. It consists of two linear layers with a ReLU activation in between.
    The first linear layer expands the dimensionality of the input from embedding_dim to hidden_dim, and the second linear layer reduces it back to embedding_dim.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim) #W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim) #W2 and b2

    def forward(self, x: Tensor) -> Tensor:
        #max(0, xW1 + b1)W2 + b2 
        #(Batch, Seq_len, Embedding_dim) --linear1--> (Batch, Seq_len, Hidden_dim) --linear2--> (Batch, Seq_len, Embedding_dim)
        x = self.linear1(x)
        x = torch.relu(x) #does max(0, x) element-wise
        x = self.dropout(x)
        x = self.linear2(x)
        return x

#Section 3.2
class MultiHeadAttention(nn.Module):
    """
    This is the multi-head attention module. It consists of multiple attention heads, each of which performs scaled dot-product attention.
    The outputs of the attention heads are concatenated and passed through a linear layer to produce the final output.
    """

    # The inputted x is applied thrice to Q (query), K (key), V (value). Each is weighted by Wq, Wk, Wv respectively.
    # Each is then split into num_heads heads along embedding dimension, and the attention is calculated for each head separately.
    # The outputs of the heads are then concatenated and passed through a linear layer to produce the final output.

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.head_dim = embedding_dim // num_heads

        #Linear layers for query, key and value
        self.query_linear = nn.Linear(embedding_dim, embedding_dim)
        self.key_linear = nn.Linear(embedding_dim, embedding_dim)
        self.value_linear = nn.Linear(embedding_dim, embedding_dim)

        #Linear layer for the output of the attention heads
        self.out_linear = nn.Linear(embedding_dim, embedding_dim)
    
    @staticmethod
    def attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None, dropout: nn.Dropout | None = None) -> Tuple[Tensor, Tensor]:
        scores = (query @ key.transpose(-2, -1)) / sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)

        if dropout is not None:
            attention_weights = dropout(attention_weights)
        
        return (attention_weights @ value), attention_weights


    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor: #mask restricts the interaction of certain words with others.
        #Apply linear layers to get query, key and value
        query = self.query_linear(q) # (Batch, Seq_len, Embedding_dim) -> (Batch, Seq_len, Embedding_dim)
        key = self.key_linear(k) # (Batch, Seq_len, Embedding_dim) -> (Batch, Seq_len, Embedding_dim)
        value = self.value_linear(v) # (Batch, Seq_len, Embedding_dim) -> (Batch, Seq_len, Embedding_dim)

        #Split the query, key and value into num_heads heads
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        x, self.attention_weights = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, embedding_dim)
        B, H, S, D = x.shape

        x = x.transpose(1, 2).contiguous().reshape(B, S, self.embedding_dim)

        return self.out_linear(x)
    

class ResidualConnection(nn.Module):
    """
    This is the residual connection module. It adds the input to the output of the sublayer and applies layer normalization.
    This is done to stabilize the training of the model and to improve the convergence.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """
    This is the encoder layer module. It consists of a multi-head attention sublayer followed by a feed forward sublayer, with residual connections around each of them.
    """

    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    """
    This is the encoder module. It consists of a stack of N encoder layers.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

class DecoderLayer(nn.Module):
    """
    This is the decoder layer module. It consists of a masked multi-head attention sublayer followed by a multi-head attention sublayer and a feed forward sublayer, with residual connections around each of them.
    """

    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x: Tensor, encoder_output: Tensor, src_mask: Tensor | None = None, tgt_mask: Tensor | None = None) -> Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    """
    This is the decoder module. It consists of a stack of N decoder layers.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x: Tensor, encoder_output: Tensor, src_mask: Tensor | None = None, tgt_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    This is the projection layer module. It projects the output of the decoder to the vocabulary size to get the logits for each word in the vocabulary.
    """

    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        return torch.log_softmax(self.linear(x), dim=-1)
    

class Transformer(nn.Module):
    """
    This is the transformer module. It consists of an encoder and a decoder, with a projection layer at the end to get the logits for each word in the vocabulary.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, tgt_embedding: InputEmbedding,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        src_embedded = self.src_embedding(src)
        src_pos_encoded = self.src_pos(src_embedded)
        return self.encoder(src_pos_encoded, src_mask)

    def decode(self, tgt: Tensor, encoder_output: Tensor, src_mask: Tensor | None = None, tgt_mask: Tensor | None = None) -> Tensor:
        tgt_embedded = self.tgt_embedding(tgt)
        tgt_pos_encoded = self.tgt_pos(tgt_embedded)
        return self.decoder(tgt_pos_encoded, encoder_output, src_mask, tgt_mask)
    
    def project(self, decoder_output: Tensor) -> Tensor:
        return self.projection_layer(decoder_output)

def buildTransformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, embedding_dim: int = 512, num_layers: int = 6, num_heads: int = 8, dropout: float = 0.1, hidden_dim: int = 2048) -> Transformer:
    src_embedding = InputEmbedding(src_vocab_size, embedding_dim)
    tgt_embedding = InputEmbedding(tgt_vocab_size, embedding_dim)

    src_pos = PositionalEncoding(src_seq_len, dropout, embedding_dim)
    tgt_pos = PositionalEncoding(tgt_seq_len, dropout, embedding_dim)

    encoder_layers = []
    for _ in range(num_layers):
        self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        feed_forward = FeedForward(embedding_dim, hidden_dim, dropout)
        encoder_layers.append(EncoderLayer(self_attention, feed_forward, dropout))
    
    decoder_layers = []
    for _ in range(num_layers):
        self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        cross_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        feed_forward = FeedForward(embedding_dim, hidden_dim, dropout)
        decoder_layers.append(DecoderLayer(self_attention, cross_attention, feed_forward, dropout))

    encoder = Encoder(nn.ModuleList(encoder_layers))
    decoder = Decoder(nn.ModuleList(decoder_layers))

    projection_layer = ProjectionLayer(embedding_dim, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection_layer)

    #Initialise the hyperparameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer