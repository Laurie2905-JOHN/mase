1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.

- Precision and Recall
Precision: The ratio of true positive predictions to the total number of positive predictions (including false positives).
Recall: The ratio of true positive predictions to the actual number of positive cases.
Usefulness: These metrics are crucial when dealing with imbalanced datasets. Precision is important when false positives are costly (email spam detection), while recall is important when false negatives carry higher risk (disease detection).

- F1 Score
Description: The F1-score metric uses a combination of precision and recall. A high F1 score high precision and recall
Usefulness: Particularly valuable in situations where it's important to find an equilibrium between precision and recall, such as in document classification or spam detection.

- Throughput
Description: Measures the number of units of work that can be processed per time unit, like samples per second during inference.
Usefulness: Important for evaluating the operational efficiency of models, especially in real-time applications.

- Energy Consumption
Description: The amount of energy required for training or inference.
Usefulness: Increasingly important for sustainable AI and in scenarios with limited power availability, like mobile devices.

- Model Size: 


2.1. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric.

GPU energy consumption was implemented although possible inaccurate (Python doesnt give an amazing way to do this). The strategy used could possible slow down cpu operations causing a change to the latency.

Precision, Recall and F1 Score were implemented to demonstrate how the output of the model changes with different configerations.

Latency was included as this gives a good understanding of how quickly the model can process and input

2.2.

When the output of your model (preds) is a probability distribution over classes, the cross_entropy loss function encourages the model to assign high probability to the correct class.

As the model gets better at this (i.e., as the loss decreases), it will more frequently assign the highest probability to the correct class, thereby increasing the accuracy.

Thus, in scenarios where the model's predictions are probabilities, and the task is to predict the class with the highest probability, a decrease in cross-entropy loss often corresponds to an increase in accuracy.

They are therefore similar (need more here)

3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.

4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.