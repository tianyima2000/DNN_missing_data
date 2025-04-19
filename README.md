This project contains code for the simulations in the paper Deep learning with missing data, and a tutorial on how to implement and train Pattern Embedded Neural Networks (PENNs).

# Tutorial
The complete code used in this tutorial can be found in the Jupyter notebook [Tutorial.ipynb](./Tutorial.ipynb). Here, we focus on how to define the class of PENNs and how to train them. The required packages for this tutorial are `torch`, `numpy`, `matplotlib` and `scikit-learn`.

The following code defines the class 

$$
\mathcal{F}_{\mathrm{PENN}} 
        &\Biggl(
        \begin{bmatrix}
        \bigl(3,\,(d, 70, 70, 70, 70)\bigr) & \\
        & \bigl(3,\,(73,70,70,70,1)\bigr) \\
        \bigl(2,\,(d,30,30,3)\bigr) & 
        \end{bmatrix}, s \Biggr),
$$

see the paper Deep learning with missing data for its definition.
Note that when we define `self.f1` below, we did not include `nn.Linear` at the end. This is because in `self.f3`, we have `nn.Linear` at the input layer, so the two linear tranformations can be combined into one linear transformation. This reduces the number of parameters to learn, and improves numeric stability.

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



