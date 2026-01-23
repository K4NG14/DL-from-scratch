import numpy as np

class Layer:
    def __init__(self, neurons, activation=None, neurons_out=None):
        self.neurons = neurons
        self.neurons_out = neurons_out

        self.z = None
        self.a = None
        self.inputs = None
        
        self.activation_name = activation
        
        if neurons_out is not None:
            self.weights = np.random.randn(neurons_out, neurons) * 0.1
            self.bias = np.zeros((neurons_out, 1))
        else:
            self.weights = None
            self.bias = None

    def activation(self, x): 
        if self.activation_name == "linear":
            return x
        elif self.activation_name == "sigmoid":    
            return 1/(1+np.exp(-x))
        elif self.activation_name == "relu":
            return np.maximum(0,x)
        elif self.activation_name == "softmax":
            x = x - np.max(x, axis=0, keepdims=True)
            exp_x = np.exp(x)
            return exp_x / np.sum(exp_x, axis=0, keepdims=True)
        else:
            return x
        
    def activation_derivative(self, grad_received):
        if self.activation_name == "relu":
            dz = grad_received * (self.z > 0).astype(float)
        elif self.activation_name == "sigmoid":
            s = 1 / (1 + np.exp(-self.z))
            dz = grad_received * s * (1 - s)
        elif self.activation_name == "softmax":
            dz = grad_received
        elif self.activation_name == "linear" or self.activation_name is None:
            dz = grad_received
        else:
            dz = grad_received

        return dz
        
    def forward_pass(self, values):
        self.inputs = values
        
        if self.weights is not None:
            self.z = self.weights @ values + self.bias
        else:
            self.z = values  
            
        self.a = self.activation(self.z)
        
        return self.a        

    def backward_pass(self, grad_received, learning_rate=0.01):
        dz = self.activation_derivative(grad_received)
        
        if self.weights is None:
            return dz  
            
        m = self.inputs.shape[1]
        dW = (1/m) * (dz @ self.inputs.T)
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        
        grad_output = self.weights.T @ dz
        
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
        
        return grad_output

    def calc_cost(self, y_true):
        m = y_true.shape[1]
        
        if self.activation_name == "softmax":
            log_probs = np.log(self.a + 1e-8)
            cost = -np.sum(y_true * log_probs) / m
        else:
            cost = np.mean((self.a - y_true) ** 2)
            
        return cost


if __name__ == "__main__":
    np.random.seed(40)  
    batch_size = 1000
    n_features = 4      
    n_classes = 2       
    n_iterations = 100
    learning_rate = 0.1
    
    centers = np.random.randn(n_features, n_classes) * 2
    labels = np.random.randint(0, n_classes, batch_size)
    X_batch = centers[:, labels] + 0.5 * np.random.randn(n_features, batch_size)

    y_one_hot = np.zeros((n_classes, batch_size))
    y_one_hot[labels, np.arange(batch_size)] = 1
   
    input_layer = Layer(neurons=n_features, neurons_out=10)
    hidden_layer = Layer(neurons=10, activation="relu", neurons_out=2)
    output_layer = Layer(neurons=2, activation="softmax", neurons_out=n_classes)
       

    print("=== Training MLP ===")
    
    
    for iteration in range(n_iterations):
        
        
        values_input = input_layer.forward_pass(X_batch)
        values_hidden = hidden_layer.forward_pass(values_input)
        output_probs = output_layer.forward_pass(values_hidden)
        
        cost = output_layer.calc_cost(y_one_hot)
        
        
        
        grad_output = output_probs - y_one_hot
        
        grad_hidden = output_layer.backward_pass(grad_output, learning_rate)
        grad_input = hidden_layer.backward_pass(grad_hidden, learning_rate)

        predictions = np.argmax(output_probs, axis=0)
        true_labels = np.argmax(y_one_hot, axis=0)
        accuracy = np.mean(predictions == true_labels) * 100

        if iteration % 10 == 0:
            print(f"\n--- Iteration {iteration }/{n_iterations} ---")
            print(f"Cost: {cost:.6f}")
            print(f"  Batch Precision: {accuracy:.2f}%")
        
        

    print("\n=== Final Prediction ===")
    
    final_input = input_layer.forward_pass(X_batch)
    final_hidden = hidden_layer.forward_pass(final_input)
    final_output = output_layer.forward_pass(final_hidden)
       
    print(f"Predictions vs True Labels:")
    final_predictions = np.argmax(final_output, axis=0)
    
    for i in range(5):
        print(f"  Example {i+1}:")
        print(f"    Probabilities: {final_output[:, i]}")
        print(f"    Class prediction: {final_predictions[i]}")
        print(f"    True class: {true_labels[i]}")
        print(f"    {'CORRECT' if final_predictions[i] == true_labels[i] else 'INCORRECT'}")

    print("...")
    
    final_accuracy = np.mean(final_predictions == true_labels) * 100
    print(f"\nFinal batch prediction: {final_accuracy:.2f}%")
    
   