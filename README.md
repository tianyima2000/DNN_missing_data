This project contains code for the simulations in the paper [Deep learning with missing data](https://arxiv.org/abs/2504.15388), and a tutorial on how to implement and train Pattern Embedded Neural Networks (PENNs).

# Tutorial
Here, we only demonstrate how to define a class of PENNs using `torch` in Python. A PENN can then be trained using stochastic gradient descent or Adam. A complete tutorial can be found in the Jupyter notebook [Tutorial.ipynb](./Tutorial.ipynb), where we train a PENN and apply a pruning-reinitialising procedure and early stopping during the training process.

Code to implement PENN using `torch`:

```python
# Define a class of Pattern Embedded Neural Networks (PENNs)
class PENN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Construct the neural network f1
        self.f1 = nn.Sequential(
            nn.Linear(d, width_1),  
            nn.ReLU(),
            nn.Linear(width_1, width_1),  
            nn.ReLU(),
            nn.Linear(width_1, width_1),  
            nn.ReLU(),
        )

        # Construct the neural network f2, i.e. the embedding function
        self.f2 = nn.Sequential(
            nn.Linear(d, width_2),  
            nn.ReLU(),
            nn.Linear(width_2, width_2),  
            nn.ReLU(),
            nn.Linear(width_2, embedding_dim)
        )

        
        # Construct the neural network f3
        self.f3 = nn.Sequential(
            nn.Linear(width_1 + embedding_dim, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, 1)  
        )
    
    # Combine f1, f2 and f3 to construct the Pattern Embedding Neural Network (PENN)
    def forward(self, z, omega):
        # compute the output of f1 and f2
        f1_output = self.f1(z)
        f2_output = self.f2(omega)
        
        # Concatenate the output of f1 and f2
        combined_features = torch.cat((f1_output, f2_output), dim=1)
        
        # Apply the combined network
        final_output = self.f3(combined_features)
        
        return final_output
