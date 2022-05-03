## COMP5214 Project

Dataset:
- train: 
  - Sampling dataset: COCO ([train2014](http://images.cocodataset.org/zips/train2014.zip)/[val2014](http://images.cocodataset.org/zips/val2014.zip))
  - Original dataset: [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- test: [Set 5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip), [Set 14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip), [Urban 100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip), [BSD 100](https://uofi.box.com/shared/static/rirohj4773jl7ef752r330rtqw23djt8.zip)

Generate datasets: `python utils.py`

Train the model: `python train.py`

