# Using neural nets to recognize handwritten digits

Neural networks are inspired by biological processes and comprise layers of artificial neurons. One of the earliest models of artificial neurons is the perceptron, developed by Frank Rosenblatt in the 1950s and 1960s. While newer models like the sigmoid neuron are often used today, understanding perceptrons lays a foundation for comprehending these more advanced neurons.

## Perceptrons

A perceptron takes binary inputs (e.g., \( x_1, x_2, . . . \)) and outputs a single binary result (0 or 1) based on the weighted sum of its inputs. The weights (\( w_1, w_2, . . . \)) indicate the importance of each input. The output is determined by whether the weighted sum exceeds a threshold:

$
\text{output} = \begin{cases}
0 & \text{if } \sum_j w_j x_j \leq \text{threshold} \\
1 & \text{if } \sum_j w_j x_j > \text{threshold}
\end{cases}
$

Thus, the perceptron functions as a decision-making model by evaluating inputs based on weighted importance.

### Example: Festival Decision-Making Using Perceptrons

Consider a decision to attend a festival based on the weather, whether a partner wants to join, and accessibility via public transit. Assigning weights (e.g., \( w_1 = 6 \) for weather, \( w_2 = 2 \) for partner, \( w_3 = 2 \) for transit) and setting a threshold allows modeling this decision. Varying the weights and threshold can yield different decision-making models, like adjusting the willingness to attend based on specific conditions.

### Perceptrons in Multi-Layered Networks

In a multi-layer network, each layer of perceptrons builds upon the previous one. The first layer makes basic decisions, while subsequent layers synthesize these results, enabling complex decision-making. In these networks, each perceptron has a single output used as inputs by other perceptrons, allowing for hierarchical decision-making processes.

### Simplifying Perceptron Notation

The perceptron condition
$\sum_j w_j x_j > \text{threshold}$
can be simplified:

- Rewrite $\sum_j w_j x_j$ as a dot product, $w \cdot x$.
- Replace the threshold with a bias $b = -\text{threshold}$, leading to:

  $
  \text{output} = \begin{cases}
  0 & \text{if } w \cdot x + b \leq 0 \\
  1 & \text{if } w \cdot x + b > 0
  \end{cases}
  $

The bias reflects how easily the perceptron outputs a 1, making further simplifications in perceptron modeling.

## Sigmoid Neurons

To enable learning in a neural network, a new kind of artificial neuron called the **sigmoid neuron** replaces the perceptron. Unlike perceptrons, which output 0 or 1 based on a threshold, sigmoid neurons can produce a range of outputs between 0 and 1. This property allows for smoother adjustments in the network's output as weights and biases change, making it easier to gradually modify and train the network.

### How Sigmoid Neurons Differ from Perceptrons

1. **Input and Output**:

   - Sigmoid neurons, like perceptrons, have inputs ($x_1, x_2, \ldots$) and corresponding weights ( $w_1, w_2, \ldots$).
   - The output is not restricted to binary values (0 or 1); instead, it is $\sigma(w \cdot x + b)$, where $\sigma$ is the **sigmoid function**.

2. **The Sigmoid Function**:

   - Defined as

     ### $\sigma(z) = \frac{1}{1 + e^{-z}}$,

     where $z = w \cdot x + b$.

   - Produces a smooth, continuous range of values between 0 and 1, unlike the step function of perceptrons. For large \( z \), the output approximates 1; for very negative \( z \), it approximates 0.

3. **Smoothness and Learning**:
   - Sigmoid neurons allow small changes in weights or biases to result in small output changes, a property essential for learning. Calculus shows that these changes are approximately **linear** with respect to changes in weights and bias.

### Why Use the Sigmoid Function?

While the sigmoid function’s exact form isn’t crucial, its **shape**—a smooth S-curve—makes it ideal for gradual adjustments. The function’s exponential form also simplifies calculations during differentiation, which is valuable when computing partial derivatives in learning algorithms.

### Interpreting Sigmoid Neuron Outputs

- **Range of Values**: Sigmoid neurons output any real number between 0 and 1, useful for representing continuous values (e.g., pixel intensities).
- **Binary Decisions**: When a binary output is needed (e.g., classifying digits), a threshold can be applied (e.g., 0.5) to interpret outputs close to 0 or 1 as desired.

## Exercises

1. Sigmoid neurons simulating perceptrons, part I
   Suppose we take all the weights and biases in a network of perceptrons, and multiply them by a positive constant, c>0. Show that the behaviour of the network doesn't change.

2. Sigmoid neurons simulating perceptrons, part II
   Suppose we have the same setup as the last problem - a network of perceptrons. Suppose also that the overall input to the network of perceptrons has been chosen. We won't need the actual input value, we just need the input to have been fixed. Suppose the weights and biases are such that w⋅x+b≠0 for the input x to any particular perceptron in the network. Now replace all the perceptrons in the network by sigmoid neurons, and multiply the weights and biases by a positive constant c>0. Show that in the limit as c→∞ the behaviour of this network of sigmoid neurons is exactly the same as the network of perceptrons. How can this fail when w⋅x+b=0 for one of the perceptrons?

## Solutions

1. Sigmoid Neurons Simulating Perceptrons, Part I

Suppose we take all weights and biases in a network of perceptrons and multiply them by a positive constant \( c > 0 \). Let's see why this doesn't change the behavior of the network.

**Perceptron Activation**: Recall that a perceptron has an output of 1 if $w \cdot x + b > 0$, and an output of 0 if $w \cdot x + b \leq 0$.

**Scaling by \( c \)**: If we multiply each weight and bias by a positive constant \( c \), then for any input \( x \), the condition for the perceptron’s output becomes:
$
    c(w \cdot x + b) > 0 \Rightarrow w \cdot x + b > 0
    $
    This is because \( c > 0 \) does not change the sign of $w \cdot x + b$.

---

2. Sigmoid Neurons Simulating Perceptrons, Part II

Now, consider a network of perceptrons where each perceptron is replaced by a sigmoid neuron. For any input \( x \), assume that \( w \cdot x + b \neq 0 \) for every perceptron in the network. Let's examine the behavior as we multiply the weights and biases by a large positive constant \( c \) and let \( c \to \infty \).

**Sigmoid Activation**: The output of a sigmoid neuron with activation $z = w \cdot x + b$ is:

$
\sigma(z) = \frac{1}{1 + e^{-z}}
$

**Scaled Activation**: When we replace \( w \) and \( b \) with \( c w \) and \( c b \), the activation becomes $z = c(w \cdot x + b)$. As $c \to \infty$:

- If $w \cdot x + b > 0$, then $c(w \cdot x + b) \to +\infty$, and thus $\sigma(c(w \cdot x + b)) \to 1$.
- If $w \cdot x + b < 0$, then $c(w \cdot x + b) \to -\infty$, and $\sigma(c(w \cdot x + b)) \to 0$.

**Behavior in the Limit**: In the limit as $c \to \infty$, the sigmoid neuron behaves exactly like a perceptron, outputting 1 when $w \cdot x + b > 0$ and 0 when $w \cdot x + b < 0$. Therefore, the network of sigmoid neurons will behave identically to the original network of perceptrons.

**Failure Condition**: If $w \cdot x + b = 0$ for some perceptron, then $c(w \cdot x + b) = 0$ regardless of \( c \). In this case, $\sigma(0) = 0.5$, which does not correspond to either 0 or 1 as in a perceptron. Thus, the sigmoid neuron’s output does not match the perceptron’s, and the behavior of the network may differ.
