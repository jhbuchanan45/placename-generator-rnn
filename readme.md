# Recurrent Neural Network for Generating Placenames

> AI is hard.

After 1 hour 30 of training, it has gone from saying random letters to saying exclusively 'a'. Nice.

On the bright side, versus the original design, it trains at up to 100x the speed by my estimation by using CUDA and
probably Tensor cores and batching, so it can learn to say exclusively 'a' in only 10 minutes. 