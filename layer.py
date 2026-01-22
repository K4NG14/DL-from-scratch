import numpy as np

class Layer:
    def __init__(self, neurons, values_received ,activation = None, neurons_out = None):
        self.neurons = neurons
        self.neurons_out = neurons_out
        self.activation_name = activation
        self.values_received = values_received
        if neurons_out is not None:
            self.weights = np.random.uniform(-1, 1, size=(neurons_out, neurons))
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
        
    def forward_pass(self):
        a = self.values_received
        if self.activation_name is not None:
            a = self.activation(a)
        return self.weights @ a + self.bias
        

if __name__ == "__main__":

    input_layer = Layer(neurons=5, values_received=np.array([[3],[4],[5],[6],[7]]),neurons_out=4)
    values = input_layer.forward_pass()
    hidden_layer = Layer(neurons=input_layer.neurons_out, values_received=values, activation="relu", neurons_out=10)
    values = hidden_layer.forward_pass()
    output_layer = Layer(neurons=hidden_layer.neurons_out, values_received=values, activation="softmax")
    output = output_layer.activation(output_layer.values_received)
    


    print("\n=== Capa de Entrada ===")
    print("Valores recibidos (input):", input_layer.values_received.flatten())
    print("Salida forward:", values.flatten()[:input_layer.neurons_out])  # salida de la capa input
    print("-"*50)

    print("\n=== Capa Oculta ===")
    print("Valores recibidos (input capa oculta):", values.flatten()[:input_layer.neurons_out])
    print("Salida ReLU:", values.flatten()[:hidden_layer.neurons_out])  # salida de capa oculta
    print("-"*50)

    print("\n=== Capa de Salida ===")
    print("Valores recibidos (input capa salida):", values.flatten()[:hidden_layer.neurons_out])
    print("Salida Softmax (probabilidades):")
    for i, val in enumerate(output.flatten(), start=1):
        print(f"  Neurona {i:2d}: {val:.4f}")
