import numpy as np
from network import Network
from activations import ReLU, Sigmoid
from dense import Dense
from losses import mse, mse_prime

np.random.seed(42)

X_train = np.linspace(-3, 3, 1000).reshape(1, 1000) 
y_train = X_train ** 2 / 9 


model = Network([
    Dense(input_size=1, output_size=2),
    Sigmoid(),
    Dense(input_size=2, output_size=1),
])

print("=== Entrenando Regresión (Aproximando x^2) ===")

model.train(
    X_train, 
    y_train, 
    loss_func=mse,        
    loss_prime=mse_prime, 
    epochs=100, 
    learning_rate=0.5,
    batch_size=32
)

x_test = np.array([[2.0], [3.0], [4.0]]).T

prediction = model.forward(x_test)

predicted_val_1 = prediction[0][0] * 9
print(f"\nEntrada: 2.0")
print(f"Valor Real (x^2): 4.0")
print(f"Predicción Red:   {predicted_val_1:.4f}")

predicted_val_2 = prediction[0][1] * 9
print(f"\nEntrada: 3.0")
print(f"Valor Real (x^2): 9.0") 
print(f"Predicción Red:   {predicted_val_2:.4f}")

predicted_val_3 = prediction[0][2] * 9
print(f"\nEntrada: 4.0")
print(f"Valor Real (x^2): 16.0") 
print(f"Predicción Red:   {predicted_val_3:.4f}")
