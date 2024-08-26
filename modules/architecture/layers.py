import torch
import torch.nn as nn

from utils import wavelet_diffusion

ACTIVATION_DICT = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(negative_slope=0.2),
    "elu": nn.ELU(),
    "gelu": nn.GELU(),
    "none": None,
    "abs": torch.abs,
    "id": lambda x: x,
}




class GNNLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            bias: bool = True,
            dropout: float = 0.,
            activation: str = "relu"
        ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = ACTIVATION_DICT[activation]


        if bias:
            self.bias = nn.Parameter(torch.zeros(1, output_dim))
        else:
            self.bias = None
        
    def forward(self, x, supp_matrix):
        
        # Dropout step
        x = self.dropout(x)

        # Transformation step
        x = self.linear(x)

        # Message passing step
        x = torch.matmul(supp_matrix, x)

        # Bias
        if self.bias is not None:
            x += self.bias

        # Activation step
        if self.activation is not None:
            x = self.activation(x)

        return x
    

class GATLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_heads: int = 8,
            bias: bool = False,
            dropout: float = 0.6,
            activation_att: str = "leaky_relu",
            activation: str = "relu",
            aggregation: str = "cat",
            skip: bool = False,
            analysis_mode: bool = False
        ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.activation = ACTIVATION_DICT[activation]
        self.activation_att = ACTIVATION_DICT[activation_att]
        self.aggregation = aggregation
        self.skip = skip
        self.analysis_mode = analysis_mode
        

        if skip:
            self.linear_skip = nn.Linear(input_dim, output_dim, bias=bias)

        self.linear = nn.Parameter(torch.empty(num_heads, input_dim, output_dim))
        nn.init.xavier_uniform_(self.linear.data)

        self.attention_vect_src = nn.Parameter(torch.empty(num_heads, output_dim, 1))
        nn.init.xavier_uniform_(self.attention_vect_src.data)

        self.attention_vect_dst = nn.Parameter(torch.empty(num_heads, output_dim, 1))
        nn.init.xavier_uniform_(self.attention_vect_dst.data)

        if bias:
            if aggregation == 'cat':
                shape = torch.Size([num_heads, 1, 1, output_dim])
            else:
                shape = torch.Size([1, 1, output_dim])
            self.bias = nn.Parameter(torch.zeros(shape))
        else:
            self.bias = None

    def forward(self, x, adj):

        # Dropout step
        x = self.dropout(x)

        # Skip connection
        if self.skip:
            x_skip = self.linear_skip(x)

        # Attention mechanism
        old_shape = x.shape
        x = x.view(-1, x.size(-1))
        h = torch.matmul(x, self.linear)
        h_src = torch.matmul(h, self.attention_vect_src)
        h_dst = torch.matmul(h, self.attention_vect_dst)

        h = h.view(h.size(0), old_shape[0], old_shape[1], -1)
        h_src = h_src.view(h_src.size(0), old_shape[0], old_shape[1], -1)
        h_dst = h_dst.view(h_dst.size(0), old_shape[0], old_shape[1], -1)

        score_mat = h_dst + h_src.transpose(-1, -2)
        if self.activation_att is not None:
            score_mat = self.activation_att(score_mat)
        score_mat = torch.where(adj > 0, score_mat, -9e15)
        att_mat = torch.softmax(score_mat, dim=-1)
        att_mat = att_mat * adj
        att_mat = self.dropout(att_mat)

        # Message passing step
        x = torch.matmul(att_mat, h)

        # Transformation step
        if self.bias is not None and self.aggregation == 'cat':
            x += self.bias

        # Aggregate attention heads
        if self.aggregation == 'cat':
            x = torch.cat([x[head] for head in range(x.size(0))], dim=-1)
        elif self.aggregation == 'mean':
            x = x.mean(0)
            if self.bias is not None and self.aggregation == 'mean':
                x += self.bias

        # Skip connection
        if self.skip:
            x = torch.cat([x, x_skip], dim=-1)

        # Activation step
        if self.activation is not None:
            x = self.activation(x)

        if self.analysis_mode:
            return x, att_mat
        else:
            return x


class HYBLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 16,
            config: list = [-1, -2, -3, 1, 2, 3],
            bias: bool = True,
            dropout: float = 0.,
            activation = 'relu',
            combination: str = 'cat'
        ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.activation = ACTIVATION_DICT[activation]
        self.combination = combination
        
        self.config = config

        self.linear_list = nn.ModuleList()
        self.bias_list = nn.ParameterList()
        for _ in range(len(config)):
            self.linear_list.append(nn.Linear(input_dim, hidden_dim, bias=False))
            if bias:
                self.bias_list.append(nn.Parameter(torch.zeros(1, hidden_dim)))

    def forward(self, x, gcn_mat, sct_mat):
        
        # Dropout step
        x = self.dropout(x)

        channel_list = []
        for i, linear in enumerate(self.linear_list):

            # Transformation step
            h = linear(x)

            # Message passing step
            scale = self.config[i]
            if scale < 0:
                supp_mat = gcn_mat
            else:
                supp_mat = sct_mat
            h = wavelet_diffusion(h, supp_mat, scale)

            # Bias
            if len(self.bias_list) > 0:
                h += self.bias_list[i]
            
            channel_list.append(h)
    
        # Combination of channels
        if self.combination == 'cat':
            x = torch.cat(channel_list, dim=-1)

        # Activation step
        if self.activation is not None:
            x = self.activation(x)

        return x


class HYBLayer_pre(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 16,
            num_channels: int = 6,
            bias: bool = True,
            dropout: float = 0.,
            activation = nn.ReLU(),
            combination: str = 'cat'
        ) -> None:
        super().__init__()

        self.activation = activation
        self.combination = combination

        self.dropout = nn.Dropout(p=dropout)
        self.linear_list = nn.ModuleList()
        self.bias_list = nn.ParameterList()
        for _ in range(num_channels):
            self.linear_list.append(nn.Linear(input_dim, hidden_dim, bias=False))
            if bias:
                self.bias_list.append(nn.Parameter(torch.zeros(1, hidden_dim)))

    def forward(self, x, supp_matrix_list):
        
        # Dropout step
        x = self.dropout(x)

        channel_list = []
        for i, linear in enumerate(self.linear_list):

            # Transformation step
            h = linear(x)

            # Message passing step
            h = torch.spmm(supp_matrix_list[i], h)

            # Bias
            if len(self.bias_list) > 0:
                h += self.bias_list[i]
            
            channel_list.append(h)
    
        # Combination of channels
        if self.combination == 'cat':
            x = torch.cat(channel_list, dim=-1)

        # Activation step
        if self.activation is not None:
            x = self.activation(x)

        return x


class RWLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 16,
            num_heads: int = 8,
            dropout: float = 0.,
            activation = nn.LeakyReLU(negative_slope=0.2)
        ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        
        self.linear = nn.Parameter(torch.empty(num_heads, input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.linear.data)

        self.attention_vect_src = nn.Parameter(torch.empty(num_heads, hidden_dim, 1))
        nn.init.xavier_uniform_(self.attention_vect_src.data)

        self.attention_vect_tar = nn.Parameter(torch.empty(num_heads, hidden_dim, 1))
        nn.init.xavier_uniform_(self.attention_vect_tar.data)

    def forward(self, x, adj):
        
        # Dropout step
        x = self.dropout(x)

        # Attention mechanism
        h = torch.matmul(x, self.linear)
        h_src = torch.matmul(h, self.attention_vect_src)
        h_tar = torch.matmul(h, self.attention_vect_tar)

        score_mat = h_tar + h_src.transpose(-1, -2)
        if self.activation is not None:
            score_mat = self.activation(score_mat)
        score_mat = torch.where(adj > 0, score_mat, -9e15)
        att_mat = torch.softmax(score_mat, dim=-2)

        return att_mat.mean(0)