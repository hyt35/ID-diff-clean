import torch.nn as nn
import torch

class deprecated_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, depth=2):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        # Adjusting the input dimension to account for the concatenation of x and t
        self.network = self.build_network(input_dim + 1, hidden_dim, depth)

    def build_network(self, input_dim, hidden_dim, depth):
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, self.input_dim))  # Ensuring output dim is same as input dim
        return nn.Sequential(*layers)

    def forward(self, x, t):
        # Concatenate x and t along the last dimension
        t = t.unsqueeze(-1)  # Ensuring t has the same batch dimension as x
        x_t = torch.cat([x, t], dim=-1)
        return self.network(x_t)

class MLP(nn.Module):
    def __init__(self, args): 
        super(MLP, self).__init__()
        state_size = args.ambient_dim  # Use ambient_dim directly
        hidden_layers = args.depth
        hidden_nodes = args.hidden_dim
        dropout = args.dropout if hasattr(args, 'dropout') else 0.0

        input_size = state_size + 1  # +1 because of the time dimension.
        output_size = state_size

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(input_size, hidden_nodes))
        self.mlp.append(nn.Dropout(dropout))  # addition
        self.mlp.append(nn.ELU())

        for _ in range(hidden_layers):
            self.mlp.append(nn.Linear(hidden_nodes, hidden_nodes))
            self.mlp.append(nn.Dropout(dropout))  # addition
            self.mlp.append(nn.ELU())
        
        self.mlp.append(nn.Linear(hidden_nodes, output_size))
        self.mlp = nn.Sequential(*self.mlp)
             
    def forward(self, x, t):
        # Concatenate x and t along the last dimension
        t = t.unsqueeze(-1)  # Ensuring t has the same batch dimension as x
        x_t = torch.cat([x, t], dim=-1)
        return self.mlp(x_t)