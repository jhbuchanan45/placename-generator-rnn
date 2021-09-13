# Recurrent Neural Network for Generating Placenames

> AI is hard.

After 1 hour 30 of training, it has gone from saying random letters to saying exclusively 'a'. Nice.

On the bright side, versus the original design, it trains at up to 100x the speed by my estimation by using CUDA and
probably Tensor cores and batching, so it can learn to say exclusively 'a' in only 10 minutes. 

Analysis of GPU usage through nvtop and nsys suggest that currently the primary bottleneck on performance is keeping
the GPU fed with processed data, as the majority of calls appear to be related to 'FillFunctor', which I presume to be
the various padding operations which are used to ensure that all placename tensors in a batch are the same length. This
is also the primary cause of the network only outputting 'a', as with randomly sorted names and fairly large batches, 
the max length tends to be significantly longer than most of the actual names. This results in the AI being fed mostly
examples of 'a' -> 'a', and it swiftly learns that the best way to get the most consistent result is to always predict 
'a', even though this doesn't hold up on individual samples.

To solve this, I intend to change how the processed data is stored, sorting it by length, with the region included 
alongside the placename. The batches will only be randomised within each length of name, so that each batch should 
include names of similar length, thus maximising the amount of useful input.

> AI is very hard

This design now trains much more effectively, and around 10% faster, however it still suffers from a tendency to bottom 
out and output all zeros. This appears to happen semi-randomly, and may be correlated with the batch size and the length
of an epoch. More research and debugging is required to determine the cause, especially pinning down if it is truly 
random or deterministically related to a certain bad sample(s). The only trained model I managed to save and sample 
without suffering 'flatlining' suffers from what appears to be severe overfitment, or something similar, and has no 
apparent grasp of regions, as the outputs are extremely similar within a few letters for any given starting letter.