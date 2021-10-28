# Privacy Lint

The Privacy Linter a library that allows you to perform a privacy analysis (Membership Inference) of your model in Pytorch. The repo implements various standard privacy attacks as well as a tool to analyze the results. With the Privacy Linter, you can:
- Run a (suboptimal) off-the-shelf analysis to approximately assess privacy leakage in your already trained model.
- Run more involved analysis to better grasp the privacy issues (for instance shadow models).
- Provide useful primitives for analysis such as grouped or balanced attacks and various metrics such as AUC or ROC.
Even if the Privacy Linter does not detect a privacy leak, it does not mean that your model is private but only that the privacy attacks fail.

The Privacy Linter will be kept up-to-date with the state-of-the-art attacks, all pull requests are welcomed!

## Usage

Below is a simple (suboptimal) example to quickly attack your `nn.Module`.

```python
from privacy_lint import LossAttack


# define and launch the attack on your model
attack = LossAttack()
results = attack.launch(model, train_loader, test_loader)

# get maximum accuracy threshold 
max_accuracy_threshold, max_accuracy = results.get_max_accuracy_threshold()
```

## Examples

See `examples/*.ipynb` for examples of:
- Shadow models attack in CIFAR-10 in `cifar10.py`.
- Loss attack on a mixture of gaussians in `Attack_GaussianClassification.ipynb`.
- Gradient attack on a mixture of gaussians in `Attack_GaussianMeanEstimation.ipynb`.
- A balanced/unbalanced attack on ImageNet in `Attack_Imagenet.ipynb`.
