Data was trained using a bash file see Run_Training_Models.sh for detail

## Modifying Batchsize

### 1.  What is the impact of varying batch sizes and why?

The batchsize of a network is a very important hyper parameter. It represents then number of samples used in one forward and backwards pass of the network. Batch sizes of 32, 64, 128, 256 and 512 will be tested, with other parameters
set as MASE default. 

![Description of image](Lab1_Results/train_acc_BS.png)
*Figure 1: Train Learning Curve with Different Batch Sizes*

![Description of image](Lab1_Results/val_acc_BS.png)
*Figure 2: Validation Learning Curve with Different Batch Sizes*

The study on how batch size affects neural network performance reveals that extreme batch sizes significantly impact learning efficiency and accuracy. Figures 1 and 2 illustrate that a small batch size of 32 leads to lower accuracy compared to other batch sizes. However, this is coupled with the drawback of extended training times. Conversely, a large batch size of 512, while substantially reducing training time, results in a decrease in both training and validation accuracy, achieving only 44.6% in validation accuracy.

An optimal batch size seems to be 256, which yields the highest validation accuracy and the second shortest training duration. Figures 1 and 2 highlight the critical role of batch size adjustment. Varying the batch size can almost halve the validation error and reduce training times by 1357%. Therefore, choosing an optimal batch size is key to neural network success.

The batch size influences training efficiency, as larger batch sizes enable more efficient use of resources like GPUs, which are optimized for parallel processing. However, excessively large batches can strain GPU memory, potentially leading to crashes or other issues.

Smaller batch sizes are often linked to better generalization since they provide noisier estimates of the gradient. This variance can sometimes hinder the learning process, but it also leads to better generalization. Studies have shown that smaller batch sizes tend to converge to flatter minima, while large batch sizes converge to sharp minima, which are associated with poorer generalization.

https://medium.com/geekculture/why-small-batch-sizes-lead-to-greater-generalization-in-deep-learning-a00a32251a4f

## Modifying Epoch Number

Epoch Number of 10, 20, 50 and 100 will be tested, using a learning rate of 0.001 with other parameters set as default.

### 1.  What is the impact of varying maximum epoch number?

![Description of image](Lab1_Results/train_acc_Epoch.png)
*Figure 3: Train Learning Curve with Different Max Epochs*

![Description of image](Lab1_Results/val_acc_Epoch.png)
*Figure 4: Validation Learning Curve with Different Max Epochs*


Figures 3 and 4 demonstrate that varying the final epoch number has a minimal effect on validation accuracy; despite slight variations, the results generally converge to a similar value. However, training times are significantly impacted. For instance, setting the epoch number to 100 leads to training times just over 3.5 hours, without any noticeable improvement in validation accuracy compared to setting it to 5 epochs, which achieves the same accuracy in only 3.7 minutes. Considering the epoch number is crucial, as excessive computation not only increases costs but also risks overfitting the training data. Given the small size of this model and its limited dataset, training for numerous epochs is unnecessary and inefficient, as shown by the results of this test.

## Modifying Learning Rate

### 3.1. What is happening with a large learning and what is happening with a small learning rate and why?

Learning rates of 0.01, 0.001, 0.0001 and 0.00001 were tested, with other parameters set as default.

![Description of image](Lab1_Results/train_acc_LR.png)
*Figure 5: Train Learning Curve with Different Learning Rates*

![Description of image](Lab1_Results/val_acc_LR.png)
*Figure 6: Validation Learning Curve with Different Learning Rates*

Figures 5 and 6 illustrate a significant change between different learning rates. Notably, a smaller learning rate converges extremely slowly and results in the final validation and training accuracy being considerably lower than with other learning rates. As the max epoch was restricted to 20, the model might reach the accuracies of other learning rates if the epoch number were increased, but then the training times would be excessive for no benefit. All training times were similar, except for the smallest learning rate, which was considered an anomaly as the others varied very little. Although training time doesn't change considerably with learning rate, a small learning rate might converge so slowly that it requires more epochs, thus increasing training time.

On the other hand, a high learning rate of 0.1 converged very quickly to a minimum, but it settled on a local minimum instead of a global one, evidenced by the high validation and training accuracy with no further convergence.

A learning rate of 10e-3 demonstrated the best performance, with both fast convergence and the highest validation accuracy. This highlights the importance of selecting a suitable learning rate to ensure convergence to a global minimum while also balancing training time.

The learning rate is crucial in determining the size of the step towards optimal weights. A large learning rate can overshoot the optimum solution, causing the optimizer to get stuck in a local minimum. Conversely, a small learning rate may lead to excessively slow convergence to an optimum. Ideally, the learning rate would continuously adapt to provide the best balance.

### 3.2. What is the relationship between learning rates and batch sizes?

![Description of image](Lab1_Results/training_acc_Large.png)
*Figure 7: Train Learning Curve with Different Batch Sizes*

![Description of image](Lab1_Results/training_acc_Large.png)
*Figure 8: Validation Learning Curve with Different Batch Sizes*

## 10x More Parameter Network {#10x-more-parameter-network}

### 4.  Implement a network that has in total around 10x more parameters than the toy network.

### 5.  Test your implementation and evaluate its performance.

![Description of image](Lab1_Results/training_acc_Large.png)
*Figure 9: Train Learning Curve with Different Batch Sizes*

![Description of image](Lab1_Results/val_acc_Large.png)
*Figure 10: Validation Learning Curve with Different Batch Sizes*

