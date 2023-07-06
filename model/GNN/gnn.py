from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from dataset import data, test_data, print_metrics


class GCN(torch.nn.Module):
    def __init__(self, input_channel, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_channel, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.prelu1 = torch.nn.PReLU()
        self.prelu2 = torch.nn.PReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv2(x, edge_index)
        # x = self.bn2(x)
        # x = self.prelu2(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv3(x, edge_index)
        # x = self.bn3(x)
        # x = torch.nn.PReLU(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        return x


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, gnn, mlp):
        super(LinkPredictor, self).__init__()
        self.gnn = gnn
        self.mlp = mlp

    def forward(self, data):
        x, future_edge_index = data.x, data.future_edge_index
        x = self.gnn(x, future_edge_index)

        # Link prediction based on concatenated node embeddings
        x_i = torch.index_select(x, 0, future_edge_index[0])
        x_j = torch.index_select(x, 0, future_edge_index[1])
        x_concat = torch.cat([x_i, x_j], dim=-1)

        return self.mlp(x_concat)


input_channel = 3
hidden_channels = 32  # dimension of node embeddings
hidden_mlp = 32  # dimension of the hidden layer in the MLP
output_dim = 1
model = LinkPredictor(
    GCN(input_channel=input_channel, hidden_channels=hidden_channels),
    MLP(input_dim=2 * hidden_channels, hidden_dim=hidden_mlp, output_dim=output_dim),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(30):
    optimizer.zero_grad()

    # Forward pass
    out = model(data)

    loss = F.binary_cross_entropy(out, data.future_edge_labels.unsqueeze(1))

    # Backward pass
    loss.backward()
    optimizer.step()

    # Print loss
    print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Evaluate on the test set
model.eval()
with torch.no_grad():
    pred = model(test_data)

print_metrics(
    test_data.future_edge_labels,
    pred,
)
