### 1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.

- Precision and Recall
Precision: The ratio of true positive predictions to the total number of positive predictions (including false positives).
Recall: The ratio of true positive predictions to the actual number of positive cases.
Usefulness: These metrics are crucial when dealing with imbalanced datasets. Precision is important when false positives are costly (email spam detection), while recall is important when false negatives carry higher risk (disease detection).

- F1 Score

Formula:

                Precision * Recall
 F1 Score = 2 * ------------------
                Precision + Recall

Description: The F1-score metric combines precision and recall, as indicated by its formula. A high F1 score signifies both high precision and recall. This metric is particularly valuable in situations where finding a balance between precision and recall is crucial, such as in document classification. While the F1 score provides a single metric for comparing classifiers, it could obscure large differences in precision or recall, as it assigns equal weight to both. In cases where it is imperative to minimize false positives or negatives, using precision and recall metrics separately may be more appropriate.

<!-- - Throughput
Description: Measures the number of units of work that can be processed per time unit, like samples per second during inference.
Usefulness: Important for evaluating the operational efficiency of models, especially in real-time applications. -->

- GPU Energy Consumption
Description: The energy required by GPUs for training or inference processes is becoming increasingly important for sustainable AI development, especially in scenarios with limited power resources like mobile devices. Large Language Models, such as OpenAI's ChatGPT, often require substantial power for training, estimated at around 10 GWh, equivalent to the average yearly consumption of 1,000 households. Additionally, with the extensive use of ChatGPT, resulting in hundreds of millions of queries, the daily energy consumption is estimated to be equivalent to 1 GWh, or the daily usage of 33,000 households. Consequently, it is crucial to make concerted efforts in reducing the energy consumption of these models to enhance the sustainability of AI technologies

https://www.washington.edu/news/2023/07/27/how-much-energy-does-chatgpt-use/

- Latency: Latency is a critical factor for neural networks during both the backward and forward passes. Measuring latency provides an estimate of the speed at which a neural network can deliver results upon receiving inputs. This aspect is crucial in numerous fields, such as high-frequency trading and autonomous driving. Latency can determine the feasibility of a model in specific scenarios. For instance, in autonomous driving, a neural network that outputs results with a delay of one second could potentially lead to a car crash. Moreover, the duration required for a model to process inputs and update weights significantly impacts training efficiency. A model that can train more rapidly not only consumes less energy but may also reach an optimal solution faster compared to one with higher latency.

### 2.1. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric.

Metrics Implemented:

- GPU energy consumption
- Precision, Recall and F1 Score 
- Latency

### 2.2.  It’s important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).

When the output of your model (predictions) represents a probability distribution across classes, the cross-entropy loss function prompts the model to assign a high probability to the correct class. As the model improves in this aspect (i.e., as the loss decreases), it increasingly assigns the highest probability to the correct class, thus enhancing the accuracy.

Therefore, in scenarios where the model's predictions are probabilities and the task involves predicting the class with the highest probability, a decrease in cross-entropy loss often correlates with an increase in accuracy. While these metrics measure different aspects, they both serve as proxies for model performance in terms of classification accuracy.

### 3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.

### 4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.