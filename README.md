### Risk Controlled Image Retrieval
Kaiwen Cai, Chris Xiaoxuan Lu, Xingyu Zhao, Wei Huang, Xiaowei Huang

### News 🔥

- 2024-12-22: Our main manuscript is available on  arXiv, and the supplementary material on [GitHub](./supplementary.pdf).

- 2024-12-22: The code for trainig and testing RCIR is released.

- 2024-12-14: Our paper has been accepted for a poster presentation at AAAI 2025!  🎉

### Citation
```
@inproceedings{cai2025risk,
    author = {Cai, Kaiwen and Lu, Chris Xiaoxuan and Zhao, Xingyu and Huang, Wei and Huang, Xiaowei},
    booktitle = {AAAI Conference on Artificial Intelligence}, 
    year = {2025},
    pages = {},
    publisher = {},
    title = {Risk Controlled Image Retrieval},
}
```

### 0. Environment

- Ubuntu 18.04
- python 3.9 + PyTorch 2.0 + CUDA 11.7

### 1. Dataset

- CUB200: https://www.vision.caltech.edu/datasets/cub_200_2011/
- CAR196: http://ai.stanford.edu/~jkrause/cars/car_dataset.html
- Pittsburgh:

  ```
  wget "https://www.dropbox.com/s/ynep8wzii1z0r6h/pittsburgh.zip?dl=0"
  unzip -q pittsburgh.zip
  ```
- ChestX-Det:

  ```
  wget "http://resource.deepwise.com/ChestX-Det/train_data.zip"
  wget "http://resource.deepwise.com/ChestX-Det/test_data.zip"
  python datasets/chestx.py   # preprocessing
  ```

  Arrange folders as follow:
  ```
  dbs
  ├── CAR196 (-cars_test, -cars_train, -devkit)
  ├── chest_x_det (-all, -train, -test, -ChestX_Det_test.json, -ChestX_Det_train.json)
  ├── CUB_200_2011 (-attributes, -images, -parts, -transformed, -bounding_boxes.txt, ..)
  └── pitts (-database, -query, -structure, -gen_test.lst, -gen_train.lst, -gen_val.lst, ..)
  ```

### 2. Train

```python
py run.py train=[model] dataset=[dataset] # `[train]=triplet|mcd|btl`, `[dataset]=cub200|car196|pitts|chestx`
e.g.,
py run.py train=triplet dataset=cub200
```

### 3. Eval

```python
py run.py train=[model] test=[model] dataset=[dataset] test.ckpt.[dataset]=[xxx.ckpt] # `[train]=[test]=triplet|mcd|btl`, `[dataset]=cub200|car196|pitts|chestx`
e.g.,
py run.py train=triplet test=triplet dataset=car196  test.ckpt.car196=logs_beta/triplet_car196_0608_223821/RCIR/tnn26715/checkpoints/best.ckpt
```

### 4. Apply RCIR

(Make sure you have completed **Eval**)

`dataset=cub200|car196|pitts|chestx`, you may need to modify the runs folder of each setting in `baselines_beta.json`, then run:

```shell
#!/bin/bash
for i in {0..9}
do
    cnt=$i
    py rcir_beta.py --unc=[model] --dbs=[dataset] --cnt=$cnt --basic & 
    # `[model]=triplet|mcd|btl|ensemble`, `[dataset]=cub200|car196|pitts|chestx`
    # e.g.,
    # py rcir_beta.py --unc=btl --dbs=car196 --cnt=$cnt --exp1 &  
done
wait
```
