### Example Usage

```
    def harmonic_oscillator_potential(x):
        return 0.5 * tf.square(x)

    potential_func = harmonic_oscillator_potential

    solver = QuantumPINNSolver(potential_func=potential_func, learning_rate=0.0001, num_epochs=1000)
    E = tf.constant(0, dtype=tf.float64)
    solver.train(E)
    solver.plot_wavefunction()
```
![A plot of the ground state solution to the quantum harmonic oscillator.](https://i.ibb.co/VmR9vcw/Figure-1.png)
