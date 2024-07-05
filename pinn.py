import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class QuantumPINNSolver:
    def __init__(self, potential_func, learning_rate=0.0001, num_epochs=1000):
        tf.keras.backend.set_floatx('float64')
        self.potential_func = potential_func
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(75, activation='relu', input_shape=(1,), dtype=tf.float64),
            tf.keras.layers.Dense(75, activation='relu', dtype=tf.float64),
            tf.keras.layers.Dense(75, activation='relu', dtype=tf.float64),
            tf.keras.layers.Dense(1, activation=None, dtype=tf.float64)
        ])
        return model

    @tf.function
    def schrodinger_loss(self, x, E):
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
        with tf.GradientTape() as tape:
            loss = self.schrodinger_loss(x, E)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, E):
        for epoch in range(self.num_epochs):
            x = tf.random.uniform((100, 1), minval=-5, maxval=5, dtype=tf.float64)
            loss = self.train_step(x, E)

            if (epoch + 1) % 100 == 0:
                tf.print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss:.4f}')

    def predict(self, x):
        return self.model(x).numpy()

    def plot_wavefunction(self, x_range=(-5, 5)):
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
        return 0.5 * tf.square(x)

    def simple_potential(x):
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
