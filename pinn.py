import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class QuantumPINNSolver:
    """
    A class to solve the quantum mechanical Schrödinger equation using Physics-Informed Neural Networks (PINNs).
    
    Attributes:
    -----------
    potential_func : function
        A function representing the potential energy function V(x).
    learning_rate : float
        The learning rate for the Adam optimizer.
    num_epochs : int
        The number of training epochs.
    model : tf.keras.Sequential
        The neural network model.
    optimizer : tf.keras.optimizers.Adam
        The optimizer used for training the neural network.
    """

    def __init__(self, potential_func, learning_rate=0.0001, num_epochs=1000):
        """
        Initializes the QuantumPINNSolver with the given potential function, learning rate, and number of epochs.

        Parameters:
        -----------
        potential_func : function
            The potential energy function V(x).
        learning_rate : float, optional
            The learning rate for the Adam optimizer. Default is 0.0001.
        num_epochs : int, optional
            The number of training epochs. Default is 1000.
        """
        tf.keras.backend.set_floatx('float64')
        self.potential_func = potential_func
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def build_model(self):
        """
        Builds the neural network model.

        Returns:
        --------
        model : tf.keras.Sequential
            The neural network model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(75, activation='relu', input_shape=(1,), dtype=tf.float64),
            tf.keras.layers.Dense(75, activation='relu', dtype=tf.float64),
            tf.keras.layers.Dense(75, activation='relu', dtype=tf.float64),
            tf.keras.layers.Dense(1, activation=None, dtype=tf.float64)
        ])
        return model

    @tf.function
    def schrodinger_loss(self, x, E):
        """
        Computes the loss based on the Schrödinger equation.

        Parameters:
        -----------
        x : tf.Tensor
            The input tensor representing the spatial coordinate.
        E : tf.Tensor
            The energy eigenvalue.

        Returns:
        --------
        loss : tf.Tensor
            The computed loss value.
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x)
                psi = self.model(x)
            dpsi_dx = tape1.gradient(psi, x)

        d2psi_dx2 = tape.gradient(dpsi_dx, x)

        print("psi:", psi)
        print("dpsi_dx:", dpsi_dx)
        print("d2psi_dx2:", d2psi_dx2)

        V_x = self.potential_func(x)
        schrodinger_eq = -d2psi_dx2 + (V_x - E) * psi
        return tf.reduce_mean(tf.square(schrodinger_eq))

    @tf.function
    def train_step(self, x, E):
        """
        Performs a single training step.

        Parameters:
        -----------
        x : tf.Tensor
            The input tensor representing the spatial coordinate.
        E : tf.Tensor
            The energy eigenvalue.

        Returns:
        --------
        loss : tf.Tensor
            The computed loss value for the current training step.
        """
        with tf.GradientTape() as tape:
            loss = self.schrodinger_loss(x, E)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, E):
        """
        Trains the model for a given number of epochs.

        Parameters:
        -----------
        E : tf.Tensor
            The energy eigenvalue.
        """
        for epoch in range(self.num_epochs):
            x = tf.random.uniform((100, 1), minval=-5, maxval=5, dtype=tf.float64)
            loss = self.train_step(x, E)

            if (epoch + 1) % 100 == 0:
                tf.print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss:.4f}')

    def predict(self, x):
        """
        Predicts the wavefunction values for the given input.

        Parameters:
        -----------
        x : tf.Tensor
            The input tensor representing the spatial coordinate.

        Returns:
        --------
        psi_x : np.ndarray
            The predicted wavefunction values.
        """
        return self.model(x).numpy()

    def plot_wavefunction(self, x_range=(-5, 5)):
        """
        Plots the wavefunction over a given range.

        Parameters:
        -----------
        x_range : tuple, optional
            The range of x values to plot. Default is (-5, 5).
        """
        x = tf.cast(tf.linspace(x_range[0], x_range[1], 100)[:, tf.newaxis], dtype=tf.float64)
        psi_x = self.predict(x)
        plt.figure(figsize=(10, 6))
        plt.plot(x, psi_x)
        plt.xlabel('x')
        plt.ylabel('psi(x)')
        plt.title('Wavefunction')
        plt.grid(True)
        plt.show()

def main():

    def harmonic_oscillator_potential(x):
        """
        Defines the harmonic oscillator potential.

        Parameters:
        -----------
        x : tf.Tensor
            The input tensor representing the spatial coordinate.

        Returns:
        --------
        V_x : tf.Tensor
            The potential energy values.
        """
        return 0.5 * tf.square(x)

    def simple_potential(x):
        """
        Defines a simple potential (zero potential).

        Parameters:
        -----------
        x : tf.Tensor
            The input tensor representing the spatial coordinate.

        Returns:
        --------
        V_x : tf.Tensor
            The potential energy values (zero).
        """
        return tf.zeros_like(x)

    # Uncomment the potential you want to use
    potential_func = harmonic_oscillator_potential
    # potential_func = simple_potential

    solver = QuantumPINNSolver(potential_func=potential_func, learning_rate=0.0001, num_epochs=1000)
    E = tf.constant(0, dtype=tf.float64)
    solver.train(E)
    solver.plot_wavefunction()

if __name__ == "__main__":
    main()
