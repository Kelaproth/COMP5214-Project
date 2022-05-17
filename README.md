## COMP5214 Project

The project is developed based on the [](), and consist of some reference to other open sourse project, including:
- IPT:
- MAE:

Dataset:
- Train: 
  - Typical Sampling dataset: COCO ([train2014](http://images.cocodataset.org/zips/train2014.zip)/[val2014](http://images.cocodataset.org/zips/val2014.zip))
  - Prepared dataset (Can also be used as Sampling): [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- Test: [Set 5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip), [Set 14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip), [Urban 100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip), [BSD 100](https://uofi.box.com/shared/static/rirohj4773jl7ef752r330rtqw23djt8.zip)

Generate dataset image list: `python utils.py`. Before that, make sure you have a folder `data`, which contains two different folder `train` and `test`, each folder contain the dataset, eg.`train2014` and only pictures in it. The tree looks like:
```
- data
    - train
        - train2014
        - div2k
    - test
        - set5
        - set14
```

Train the model: `python train.py`. The settings of the training and the model can be 
changed at the head of the train.


Vit: Use [pre-trained IPT model](https://drive.google.com/file/d/1_NN-fr3NWwNzLvj_2S5Hdf2KgeYZVIXz/view?usp=sharing), run `python .\main_ipt.py` to evaluate.

