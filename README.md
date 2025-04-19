This project contains code for the simulations in the paper Deep learning with missing data, and a tutorial on how to implement and train Pattern Embedded Neural Networks (PENNs).

# Tutorial
The complete code used in this tutorial can be found in the Jupyter notebook [Tutorial.ipynb](./Tutorial.ipynb). Here, we only focus on how to define the class of PENNs and how to train them. The required packages for this tutorial are `torch`, `numpy`, `matplotlib` and `scikit-learn`.

The following code defines the class 

```python
# Define the class of Pattern Embedded Neural Networks (PENN)
class PENN(nn.Module):
    def __init__(self):
        super().__init__()
        embedding_dim = 3
        
        # Construct the neural network f1
        self.f1 = nn.Sequential(
            nn.Linear(d, 70),  
            nn.ReLU(),
            nn.Linear(70, 70), 
            nn.ReLU(),
            nn.Linear(70, 70), 
            nn.ReLU()
        )

        # Construct the neural network f2, i.e. the embedding function
        self.f2 = nn.Sequential(
            nn.Linear(d, 30),  
            nn.ReLU(),
            nn.Linear(30, 30),  
            nn.ReLU(),
            nn.Linear(30, embedding_dim)
        )

        
        # Construct the neural network f3
        self.f3 = nn.Sequential(
            nn.Linear(70 + embedding_dim, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 1)  
        )
    
    # Combine f1, f2 and f3 to construct the Pattern Embedded Neural Network
    def forward(self, z, omega):
        # Compute the output of f1 and f2
        f1_output = self.f1(z)
        f2_output = self.f2(omega)
        
        # Concatenate the output of f1 and f2
        combined_features = torch.cat((f1_output, f2_output), dim=1)
        
        # Apply f3 to the combined output
        final_output = self.f3(combined_features)
        
        return final_output
```



