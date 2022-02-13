# TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting

Long-term time series forecasting plays an important role in numerous fields.Various deep learning models, such as recurrent neural networks and Transformer-based approaches, have shown great success in multi-horizon forecasting. However, given the challenging nature of long term forecasting, these methods tend to introduce higher model complexity to handle difficult examples in long term forecasting, which often leads to a significant increase in computation and less robustness in performance (e.g. overfitting). In this work, we propose a novel neural network architecture, termed TreeDRNet, for more effective long term forecasting. Inspired by robust regression, we introduce doubly residual link structure to make prediction more robust. Built upon Kolmogorov–Arnold representation theorem, we explicitly introduce feature selection, model ensemble, and a tree structure to further improve the robustness and representation power of TreeDRNet. Unlike previous work of deep model for sequential forecasting, TreeDRNet is built entirely based on MLP, and thus enjoys high computational efficiency. Our extensive empirical studies show that TreeDRNet is significantly more effective than state-of-the-art methods, reducing prediction errors by $20\%$ to $40\%$ for multivariate time series. In addition, TreeDRNet is over $10$ times more efficient than Transformer-based methods. 


## Long Horizon Datasets Results


### Run TreeDRNet experiment from console

To replicate the results of the paper, in particular to produce the forecasts for , run the following:


CUDA_VISIBLE_DEVICES=0 python -m treeDRNet_multivariate --hyperopt_max_evals 1 --experiment_id run_1


###  Evaluate results for a dataset using:

python -m evaluation --dataset ETTm2 --horizon -1 --model treeDRNet --experiment run_1





