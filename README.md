# 🧠 Linear Regression using TensorFlow: A Beginner-Friendly Walkthrough

This project demonstrates how to build a simple linear regression model using TensorFlow in a way that's accessible to complete beginners. It walks through synthetic data creation, model building, training, and visualization.

---

## 📁 Project Structure

```
TensorFlow-Linear-Model/
│
├── TensorFlow.ipynb      # Main Jupyter notebook with code
├── TF_Intro.npz          # Dataset generated for training
├── README.md             # This file
```

---

## 🧰 Requirements

- Python 3.10
- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter Notebook

You can install the requirements with:

```bash
pip install numpy matplotlib tensorflow
```

---

## 📖 Code Walkthrough

### 🔹 1. Importing Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
```

> We import tools for numerical operations (NumPy), plotting (Matplotlib), and TensorFlow (in version 1 compatibility mode). Disabling eager execution helps us use `placeholder`-style graphs from TensorFlow 1.x.

---

### 🔹 2. Generating Synthetic Data

```python
observations = 1000
x = np.random.uniform(-10, 10, (observations, 1))
z = np.random.uniform(-10, 10, (observations, 1))

generated_inputs = np.column_stack((x, z))

noise = np.random.uniform(-1, 1, (observations, 1))
generated_targets = 4 * x - 2 * z + 7 + noise

np.savez('TF_Intro', inputs=generated_inputs, targets=generated_targets)
```

> We generate 1000 samples using the formula:  
> `target = 4 * x - 2 * z + 7 + noise`,  
> simulating real-world data with a little randomness.

---

### 🔹 3. Setting Input and Output Dimensions

```python
input_size = 2
output_size = 1

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])
```

> `input_size` is 2 (because we have x and z).  
> `output_size` is 1 (we predict a single target).  
> Placeholders are used to feed data into the model during training.

---

### 🔹 4. Initializing Weights and Biases

```python
weights = tf.Variable(tf.random_uniform([input_size, output_size], minval=-0.1, maxval=0.1))
biases = tf.Variable(tf.random_uniform([output_size], minval=-0.1, maxval=0.1))
```

> The model starts with random values for weights and biases, which are refined during training to make better predictions.

---

### 🔹 5. Model Output Equation

```python
outputs = tf.matmul(inputs, weights) + biases
```

> This is the basic formula for linear regression:  
> **output = input × weight + bias**

---

### 🔹 6. Defining the Loss Function

```python
loss = tf.reduce_mean(tf.square(outputs - targets))
```

> We use **Mean Squared Error (MSE)** to measure how far off our predictions are from the actual targets.

---

### 🔹 7. Optimizer Setup

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
training_op = optimizer.minimize(loss)
```

> We use **gradient descent** to minimize the loss function by adjusting weights and biases iteratively.

---

### 🔹 8. Training the Model

```python
session = tf.Session()
session.run(tf.global_variables_initializer())

for epoch in range(100):
    _, curr_loss = session.run([training_op, loss], feed_dict={inputs: train_inputs, targets: train_targets})
    print(f'Epoch {epoch} - Loss: {curr_loss}')
```

> We train the model over 100 passes (epochs), each time feeding the data and letting TensorFlow update the weights to reduce the error.

---

### 🔹 9. Visualizing the Predictions

```python
predictions = session.run(outputs, feed_dict={inputs: train_inputs})

plt.plot(predictions, label='Predictions')
plt.plot(train_targets, label='Targets')
plt.legend()
plt.show()
```

> A simple plot to visually compare the model's predictions with the actual target values.

---

## 📈 What You'll Learn

- How to simulate real-world data for machine learning
- TensorFlow's computation graph concept using placeholders
- How gradient descent helps improve model performance
- Visualizing model predictions vs actual targets

---

## 📃 License

This project is licensed under the MIT License. Feel free to use or modify it for your learning or personal projects.

---


