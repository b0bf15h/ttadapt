### Acknowledgements
This repository is largely adpated from the work by https://github.com/mariodoebler/test-time-adaptation. Please refer to their well-documented README to setup the repository. 

### Run Experiments
To run experiments,
+ Specify the output directory, TTA setting (continuous, correlated, etc...) , number of corruption samples in conf.py
+ Specify other configurations in ./cfgs/[dataset]/[method.yaml]
+ Specify the data for adversarial robustness evaluation in test_time.py. For example, when adapting to CIFAR10-C, uncomment "data = load_cifar10(n_examples = 100)" and comment out "data = load_cifar100(n_examples = 100)"
+ Run python test_time.py --cfg cfgs/[cifar10_c/cifar100_c]/[source/norm_test/tent/rdumb/rotta/roid].yaml

After running the python script, you should find in the output directory, a **.txt** file which contains logs of the out-of-distribution performance when adapting to corruptions and a **.json** file which contains the model's adversarial accuracy against attacks of various budget.

