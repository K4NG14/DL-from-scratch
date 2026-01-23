import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mlp import MLP

def visualize_network(nn, X, y, costs):
    """Visualize neural network performance and decision boundaries."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Identify layers with weights for visualization
    layers_with_weights = [l for l in nn.layers if l.weights is not None]
    n_layers = len(layers_with_weights)
    
    # Setup figure with dynamic grid
    n_rows = max(2, n_layers)
    fig = plt.figure(figsize=(15, 4 * n_rows))
    gs = gridspec.GridSpec(2, 1, figure=fig)

    # Learning Curve
    ax_cost = fig.add_subplot(gs[0, 0:2])
    ax_cost.plot(costs, color='#E74C3C', linewidth=2)
    ax_cost.set_title("Learning Curve", fontsize=12, fontweight='bold')
    ax_cost.set_ylabel("Cost")

    # Decision Boundary
    ax_boundary = fig.add_subplot(gs[1, 0:2])
    
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    probs = nn.forward_pass(grid_points)
    Z = np.argmax(probs, axis=0).reshape(xx.shape)

    ax_boundary.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    true_labels = np.argmax(y, axis=0)
    scatter = ax_boundary.scatter(
        X[0, :], X[1, :], 
        c=true_labels, 
        edgecolors='k', 
        cmap='viridis', 
        s=50
    )
    ax_boundary.set_title("Decision Boundary", fontsize=12, fontweight='bold')
    ax_boundary.legend(*scatter.legend_elements(), title="Classes")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate synthetic data for visualization
    n_samples = 1000
    n_features = 2  # Must be 2 for 2D visualization
    n_classes = 4
    
    # Create Gaussian clusters
    centers = np.random.randn(n_features, n_classes) * 4
    labels = np.random.randint(0, n_classes, n_samples)
    X = centers[:, labels] + 0.6 * np.random.randn(n_features, n_samples)

    # One-hot encode labels
    y = np.zeros((n_classes, n_samples))
    y[labels, np.arange(n_samples)] = 1

    # Define network architecture
    hidden_layers = [8, 3, n_classes]
    activations = ["sigmoid", "sigmoid"]
    nn = MLP(features=n_features, classes=n_classes,activations=activations, hidden_layers=hidden_layers)

    print("Training for visualization...")
    costs = nn.train(X, y, iterations=25000, learning_rate=0.1)

    print("Generating plots...")
    visualize_network(nn, X, y, costs)