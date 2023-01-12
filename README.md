# IRNet: A Robust Deep Model for Long Term Time Series Forecasting

Various deep learning models, especially some latest Transformer-based approaches, have greatly improved the state-of-art performance for long-term time series forecasting. However, those Transformer based models suffer a severe deterioration performance with prolonged input length, which prohibits them from using extended historical information.  Moreover, these methods tend to handle complex examples in long-term forecasting with increased model complexity, which often leads to a significant increase in computation and less robustness in performance (e.g., overfitting). We propose a novel neural network architecture, called IRNet, for more effective long-term forecasting. Inspired by robust regression, we introduce a gated doubly residual link structure to make prediction more robust. Built upon Kolmogorov–Arnold representation theorem, we explicitly introduce feature selection, model ensemble, and a tree structure to further utilize the extended input sequence, which improves the robustness and representation power of IRNet. Compared with popular deep models for sequential forecasting, IRNet is built entirely on multilayer perceptron and thus enjoys high computational efficiency. Our extensive empirical studies show that IRNet reduces prediction errors by 18.3\% for multivariate time series. In particular, IRNet is significantly more effective than state-of-the-art methods, and is over $10$ times more efficient than Transformer-based methods.


## Long Horizon Datasets Results


### Run IRNet experiment from console

To replicate the results of the paper, in particular to produce the forecasts for , run the following:


CUDA_VISIBLE_DEVICES=0 python -m IRNet_multivariate --hyperopt_max_evals 1 --experiment_id run_1


###  Evaluate results for a dataset using:

python -m evaluation --dataset ETTm2 --horizon -1 --model IRNet --experiment run_1





