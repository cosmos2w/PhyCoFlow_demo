import torch
import torch.nn as nn
from .fcn import FCN

class MIMONet(nn.Module):
    def __init__(self, branch_arch_list, trunk_arch, num_outputs=1, activation_fn=nn.ReLU, merge_type='mul'):
        """
        Args:
            branch_arch_list (list of lists): List of architectures for each branch input.
            trunk_arch (list): Architecture for the trunk network.
            activation_fn (torch.nn.Module): Activation function to use in the networks.
            num_outputs (int): Number of output dimensions per trunk point.
            merge_type (str): Method to combine branch outputs ('sum' or 'mul').
        """
        super(MIMONet, self).__init__()

        self.merge_type = merge_type
        self.num_outputs = num_outputs
        self.hidden_dim = branch_arch_list[0][-1]  # final dim of each branch output

        # Branch networks
        self.branch_nets = nn.ModuleList([
            FCN(arch, activation_fn) for arch in branch_arch_list
        ])

        # Trunk network: outputs hidden_dim * num_outputs
        trunk_arch[-1] = self.hidden_dim * self.num_outputs
        self.trunk_net = FCN(trunk_arch, activation_fn)

        # Optional output bias (broadcastable across batch & points)
        self.bias = nn.Parameter(torch.zeros(1, 1, num_outputs))

    def forward(self, branch_inputs, trunk_input):
        """
        Forward pass.

        Args:
            branch_inputs (list[Tensor]): Each tensor is (batch_size, input_dim)
            trunk_input (Tensor): (batch_size, num_trunk_points, input_dim)

        Returns:
            Tensor: (batch_size, num_trunk_points, num_outputs)
        """
        # Branch processing: each output is (batch, hidden_dim)
        branch_outputs = [net(inp) for net, inp in zip(self.branch_nets, branch_inputs)]

        # Merge branches
        if self.merge_type == 'sum':
            combined_branch = sum(branch_outputs)
        elif self.merge_type == 'mul':
            combined_branch = branch_outputs[0]
            for b in branch_outputs[1:]:
                combined_branch = combined_branch * b
        else:
            raise ValueError(f"Unsupported merge type: {self.merge_type}")

        # Trunk processing
        trunk_out = self.trunk_net(trunk_input)  # (batch, num_trunk_points, hidden_dim * num_outputs)
        B, P = trunk_out.shape[0], trunk_out.shape[1]
        trunk_out = trunk_out.view(B, P, self.hidden_dim, self.num_outputs)  # (B, P, I, O)

        # Einsum: (B, I) × (B, P, I, O) → (B, P, O)
        output = torch.einsum('bi,bpio->bpo', combined_branch, trunk_out)
        return output + self.bias  # final shape: (B, P, O)

class MIMONet_Transform(nn.Module):
    def __init__(self, branch_arch_list, trunk_arch, num_outputs=1, activation_fn=nn.ReLU, merge_type='mul', output_transform=True):
        """
        Args:
            branch_arch_list (list of lists): List of architectures for each branch input.
            trunk_arch (list): Architecture for the trunk network.
            activation_fn (torch.nn.Module): Activation function to use in the networks.
            num_outputs (int): Number of output dimensions per trunk point.
            merge_type (str): Method to combine branch outputs ('sum' or 'mul').
            output_transform (bool): Whether to apply output transformation using normalization + MLP.
        """
        super(MIMONet_Transform, self).__init__()

        self.merge_type = merge_type
        self.num_outputs = num_outputs
        self.hidden_dim = branch_arch_list[0][-1]

        # Branch networks
        self.branch_nets = nn.ModuleList([
            FCN(arch, activation_fn) for arch in branch_arch_list
        ])

        # Trunk network: final output dim is hidden_dim * num_outputs
        trunk_arch[-1] = self.hidden_dim * self.num_outputs
        self.trunk_net = FCN(trunk_arch, activation_fn)

        # Optional bias
        self.bias = nn.Parameter(torch.zeros(1, 1, num_outputs))

        # Output transformation module
        self.output_transform = output_transform
        if output_transform:
            self.output_adapter = nn.Sequential(
                nn.LayerNorm(num_outputs),
                nn.Linear(num_outputs, num_outputs),
                nn.GELU(),
                nn.Linear(num_outputs, num_outputs),
                nn.Tanh(),
            )

    def forward(self, branch_inputs, trunk_input):
        """
        Args:
            branch_inputs (list[Tensor]): List of tensors with shape (B, D_i)
            trunk_input (Tensor): Tensor of shape (B, P, D_t)

        Returns:
            Tensor: (B, P, num_outputs)
        """
        # Process branches
        branch_outputs = [net(inp) for net, inp in zip(self.branch_nets, branch_inputs)]

        if self.merge_type == 'sum':
            combined_branch = sum(branch_outputs)
        elif self.merge_type == 'mul':
            combined_branch = branch_outputs[0]
            for b in branch_outputs[1:]:
                combined_branch *= b
        else:
            raise ValueError(f"Unsupported merge type: {self.merge_type}")

        # Trunk network
        trunk_out = self.trunk_net(trunk_input)  # (B, P, hidden_dim * num_outputs)
        B, P = trunk_out.shape[0], trunk_out.shape[1]
        trunk_out = trunk_out.view(B, P, self.hidden_dim, self.num_outputs)

        # Merge branches and trunk: einsum
        output = torch.einsum('bi,bpio->bpo', combined_branch, trunk_out)

        # Apply nonlinear output adapter if enabled
        if self.output_transform:
            output = output + self.output_adapter(output)

        return output + self.bias
