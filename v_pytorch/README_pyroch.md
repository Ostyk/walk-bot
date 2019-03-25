
* [PyTorch implementation](v_pytorch): taken from Move37 and altered to make project

* Running:
Training
```
python3 ppo_train.py -n [NameOfRun]
```

Testing
```
python3 ppo_play.py -m [ModelName] -e [EnvName] -d  -r [path_to_model]
```

optional:

- -e : EnvName 
- -d : deterministic 
- -r : record 

