Neural Network implementations

See: https://www.hijerry.cn/p/53364.html

## Requirements

* numpy 1.16.5
* pandas 0.25.1
* matplotlib 3.1.1
* scikit-learn 0.21.2  (as a comparison) 

## Usages

### Process Oriented

`ann.py` is process oriented implementation of Single layer Feedforward Neural Network (单层前馈神经网络)

For test, just run it:

```shell
python ann.py

# outputs:
# Running ann tests...
# Success!!
```

### Object Oriented

`NN.py` is object oriented implementation of Neural Network

Usage of this approche can be found in main.py



### Iris classification

We can get loss curve by running:

```
python main.py
```

![](http://static.hijerry.cn/affa99ce171743239e6decb29af70f30.jpg)

Red one is our approches, black one is scikit-learn approches. 

