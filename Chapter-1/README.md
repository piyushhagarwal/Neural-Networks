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
