### Why is it unusual to sequence three linear layers consecutively without nonlinear activation functions?

When multiple linear layers are stacked together without any nonlinear activations in between, their combined effect is equivalent to that of a single linear layer. Mathematically, if you have three linear layers defined as L_1, L_2, and L_3 applying them consecutively to an input with no nonlinear activation, these layers could be represented as one L_4 layer. This does not add any additional complexity or learning capacity to the model beyond what a single linear layer could achieve.

### 1. Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the ReLU also.

### 2. In lab3, we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?

### 3. You may have noticed, one problem with the channel multiplier is that it scales all layers uniformly, ideally, we would like to be able to construct networks like the following:

### 4.

The deepcopy function is designed to create a new compound object, into which it recursively inserts copies of the objects found in the original. In contrast, a shallow copy creates a new compound object but inserts references to the original objects to the extent possible. More details can be found in the Python documentation: https://docs.python.org/3/library/copy.html.

The need for deepcopy arose during a process where the graph was to be amended by the specified channel multiplier. However, using shallow copy resulted in the layers being multiplied multiple times, such as 16 x 2 in the first loop, then leading to 32 x 2 in the next..., which would cause a dimension error.

However, using deepcopy caused an error:

RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.  If you were attempting to deepcopy a module, this may be because of a torch.nn.utils.weight_norm usage, see https://github.com/pytorch/pytorch/pull/103001

Therefore, using a workaround the redefine_linear_transform_pass function was adapted and placed in a new file. See 