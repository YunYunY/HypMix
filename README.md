# Official code for [Understanding Hyperbolic Metric Learning through Hard Negative Sampling](https://openaccess.thecvf.com/content/WACV2024/papers/Yue_Understanding_Hyperbolic_Metric_Learning_Through_Hard_Negative_Sampling_WACV_2024_paper.pdf)

## This code is adapted from [Hyperbolic Vision Transformers: Combining Improvements in Metric Learning](https://github.com/htdt/hyp_metric) 

### Loss is included in 
- `train.py` - main training and contrastive_loss;

### Environment
- Python 3.7.16
- System: Linux
- Our code works on single A100
- Packages are same as the baseline method

### Run training
python -m train --ds CUB --model vit_small_patch16_224 --num_samples 9 --lr 3e-5 --ep 50 --eval_ep "[50]" --resize 256 --seed 42 --hyp_c 0.1 --t_e 0.05 --t_h 0.2 --lam 3
### Download Datasets
- [CUB-200](http://www.vision.caltech.edu/datasets/cub_200_2011/)
- Stanford Online Products ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
- [Cars-196](https://paperswithcode.com/dataset/cars196)
  


  

