import numpy as np
from layer import Layer

class  MLP:
    def __init__(self, features, classes, hidden_layers, activations = None):
        self.features = features
        self.classes = classes
        self.hidden_layers = hidden_layers
        self.activation = activations

        if self.activation is not None:
            if len(self.activation) != len(hidden_layers) - 1:
                print("num of activations does not match with num of hidden layers. activations = hidden layers - 1")
                self.activation = None
        
        self.layers = []
            
        self.layers.append(Layer(neurons=features, neurons_out=hidden_layers[0]))
        
        for i in range(len(hidden_layers) - 1):
            if self.activation == None:
                self.layers.append(Layer(
                neurons=hidden_layers[i], 
                activation="relu", 
                neurons_out=hidden_layers[i + 1]
                ))
            else:
                self.layers.append(Layer(
                neurons=hidden_layers[i], 
                activation=self.activation[i], 
                neurons_out=hidden_layers[i + 1]
                ))
            
        
        output_layer = hidden_layers[-1]
        self.layers.append(Layer(
            neurons=output_layer, 
            activation="softmax", 
            neurons_out=classes
        ))
    
    def forward_pass(self, X):
        values = X
        for layer in self.layers:
            values = layer.forward_pass(values)
        return values
    
    def backward_pass(self, y_true, learning_rate=0.01):
        output = self.layers[-1].a
        grad_output = output - y_true
        
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad, learning_rate)
            
    
    def calc_cost(self, y_true):
        return self.layers[-1].calc_cost(y_true)
    
    def predict(self, X):
        probs = self.forward_pass(X)
        return np.argmax(probs, axis=0)
    
    def train(self, X, y, iterations=100, learning_rate=0.01):
        costs = []
        
        for i in range(iterations):

            _ = self.forward_pass(X)
            
            cost = self.calc_cost(y)
            costs.append(cost)
            
            self.backward_pass(y, learning_rate)
            
            if i % 100 == 0:
                preds = self.predict(X)
                true_labels = np.argmax(y, axis=0)
                accuracy = np.mean(preds == true_labels) * 100
                print(f"Iteration {i}: Cost={cost:.4f}, Accuracy={accuracy:.1f}%")
        
        return costs
    

if __name__ == "__main__":

    np.random.seed(42)
    n_samples = 100
    n_features = 5
    n_classes = 4
    

    centers = np.random.randn(n_features, n_classes) 
    labels = np.random.randint(0, n_classes, n_samples)
    X = centers[:, labels] + 0.5 * np.random.randn(n_features, n_samples)

    y = np.zeros((n_classes, n_samples))
    y[labels, np.arange(n_samples)] = 1

    
    
    hidden_layers = [10, 3, 4]  # 3 hidden layers
    nn = MLP(
        features=n_features,
        classes=n_classes,
        hidden_layers=hidden_layers
    )
    
    print("Network:")
    for i, layer in enumerate(nn.layers):
        act = layer.activation_name 
        print(f"Layer {i}: {layer.neurons} → {layer.neurons_out} ({act})")
    
    print("\nTraining...")
    costs = nn.train(X, y, iterations=1000, learning_rate=0.1)
    
    print(f"\nInitial cost: {costs[0]:.4f}")
    print(f"Final cost: {costs[-1]:.4f}")
        