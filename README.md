## Project Description

Our project introduces *Parallel-mentoring*, a novel and effective method for offline model-based optimization. The aim is to maximize a black-box objective function using a static dataset of designs and scores across various domains. This method facilitates mentoring among proxies, creating a more robust ensemble to mitigate the out-of-distribution issue. 

We focus on the three-proxy case and instantiate *parallel-mentoring* as *tri-mentoring* with two modules: *voting-based pairwise supervision* and *adaptive soft-labeling*.

## Installation
This project relies heavily on the following key libraries:

- pytorch
- design-bench

You can install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```


## Reproducing Performance

You can reproduce the performance for the TF Bind 8 task by running our method as follows:
```bash
python main.py --task TFBind8-Exact-v0 --majority_voting 1 --soft_label 1
```
To run our method without *voting-based pairwise supervision* as:
```bash
python main.py --task TFBind8-Exact-v0 --majority_voting 0 --soft_label 1
```

To run our method without *adaptive soft-labeling* as:
```bash
python main.py --task TFBind8-Exact-v0 --majority_voting 1 --soft_label 0
```

The same commands apply for the AntMorphology task:
```bash
python main.py --task AntMorphology-Exact-v0 --majority_voting 1 --soft_label 1
python main.py --task AntMorphology-Exact-v0 --majority_voting 0 --soft_label 1
python main.py --task AntMorphology-Exact-v0 --majority_voting 1 --soft_label 0

```
## Hyperparameter Sensitivity 

You can evaluate the influence of $K$ on the model's performance by running the commands below. This will execute the model with different $K$ values (5, 10, 15, 20, 25) for the TFBind8 and AntMorphology tasks.

```bash
K_values=(5 10 15 20 25)
for K in ${K_values[*]};
do
    python main.py --task TFBind8-Exact-v0 --K $K
    python main.py --task AntMorphology-Exact-v0 --K $K
done
```

To examine the sensitivity of the model to the number of optimization steps ($T$), you can set $T$ as large as 400 and run the model again for both tasks:

```bash
python main.py --task TFBind8-Exact-v0 --Tmax 400
python main.py --task AntMorphology-Exact-v0 --Tmax 400
```

## Acknowledgements
We extend our appreciation to the design-bench library (https://github.com/brandontrabucco/design-bench) for their invaluable resources.
