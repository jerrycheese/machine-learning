Book recommendation implemantation:
* Collaborative Filter (Item based)
* Matrix Factorization

Requirements:

* Tensorflow 1.14
* numpy 1.16.5
* pandas 0.25.1
* matplotlib 3.1.1

## Usage

run:

```
python main.py --train=data.csv --method=cf
```

`--method` to choose whether  `cf` (Collaborative Filter) or `mf` (Matrix Factorization)

It may cost > 10 min to predict all `no rating` data in data.csv, and then output the top `k=10` unreaded books for user `pred_user = 140756691` (see [main.py#L125](main.py#L125))

for `mf` method, It is hard to make prediction for all book, so we predict the rating of first 3 user to first 5 book (see [Recommendation.py#L250](Recommendation.py#L250))

