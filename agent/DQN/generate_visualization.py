import torch

from DQN.Agent import DeepQNetwork


# Assuming your input dimensions (for example, 10 features and batch size of 1)
input_dims = 22
batch_size = 512
sequence_length = 30  # as per your Agent class
hidden_dims = 512
n_actions = 4  # number of actions, change as per your requirement

# Initialize your network
model = DeepQNetwork(lr=0.001, feature_count=input_dims, hidden_dims=hidden_dims, n_actions=n_actions)

model.load_state_dict(torch.load('models_bak/rac1_hoverboard_44382_600.pt'))
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_weights(model):
    for name, param in model.named_parameters():
        if 'fc' in name and 'weight' in name:
            plt.figure(figsize=(10, 5))
            sns.heatmap(param.detach().cpu().numpy(), cmap='viridis')
            plt.title(f'Heatmap of {name}')
            plt.xlabel('Output Neurons')
            plt.ylabel('Input Neurons')
            plt.show()


def visualize_lstm_weights(model):
    for name, param in model.named_parameters():
        if 'lstm.weight' in name:
            # LSTM weights are in the order: [input, forget, cell, output]
            # Each gate has two parts: weights for input features (Wi, Wf, Wc, Wo)
            # and weights for hidden states (Ui, Uf, Uc, Uo)
            all_gates = param.detach().cpu().numpy()
            input_size = model.lstm.input_size
            hidden_size = model.lstm.hidden_size

            # Split the weights for each gate
            Wi = all_gates[:hidden_size, :input_size]  # Input gate weights
            Wf = all_gates[hidden_size:2*hidden_size, :input_size]  # Forget gate weights
            Wc = all_gates[2*hidden_size:3*hidden_size, :input_size]  # Cell gate weights
            Wo = all_gates[3*hidden_size:, :input_size]  # Output gate weights

            # Plotting
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            sns.heatmap(Wi, ax=axs[0, 0], cmap='viridis')
            axs[0, 0].set_title('Input Gate Weights')
            sns.heatmap(Wf, ax=axs[0, 1], cmap='viridis')
            axs[0, 1].set_title('Forget Gate Weights')
            sns.heatmap(Wc, ax=axs[1, 0], cmap='viridis')
            axs[1, 0].set_title('Cell Gate Weights')
            sns.heatmap(Wo, ax=axs[1, 1], cmap='viridis')
            axs[1, 1].set_title('Output Gate Weights')

            plt.show()


visualize_lstm_weights(model)
visualize_weights(model)

# dummy_input = torch.rand(4, sequence_length, input_dims).to(device)
#
# with torch.no_grad():
#     _, attention_weights = model(dummy_input, attention=True)
#
# # Convert to numpy for visualization
# attention_weights = attention_weights.cpu().numpy()
#
# # Now you can visualize the attention weights
# plt.figure(figsize=(12, 6))
# sns.heatmap(attention_weights, cmap='viridis', annot=True)
# plt.title('Attention Weights')
# plt.xlabel('Timesteps')
# plt.ylabel('Batch Instances')
# plt.show()

# # Create a dummy input tensor (adjust the shape as per your network's input)
# dummy_input = torch.rand(batch_size, sequence_length, input_dims).to(device)
#
# # Perform a forward pass (just for visualization purposes)
# output = model(dummy_input)
#
# # Create the visual graph
# vis_graph = make_dot(output, params=dict(list(model.named_parameters())))

# Save the graph as a pdf or display it
#vis_graph.render('network_visualization', format='pdf')
