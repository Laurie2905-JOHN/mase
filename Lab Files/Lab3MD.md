1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.

- Precision and Recall
Precision: The ratio of true positive predictions to the total number of positive predictions (including false positives).
Recall: The ratio of true positive predictions to the actual number of positive cases.
Usefulness: These metrics are crucial when dealing with imbalanced datasets. Precision is important when false positives are costly, while recall is important when false negatives carry higher risk.
- F1 Score
Description: The harmonic mean of precision and recall. It provides a single metric that balances both the concerns of precision and recall.
Usefulness: Particularly valuable in situations where it's important to find an equilibrium between precision and recall, such as in document classification or spam detection.
- Throughput
Description: Measures the number of units of work that can be processed per time unit, like samples per second during inference.
Usefulness: Important for evaluating the operational efficiency of models, especially in real-time applications.
- Energy Consumption
Description: The amount of energy required for training or inference.
Usefulness: Increasingly important for sustainable AI and in scenarios with limited power availability, like mobile devices.


2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It’s important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).

3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.

4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.