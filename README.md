## StyleGAN2-ADA &mdash; for Audio Textures 

This forked repository has updates for modelling Audio Textures. Please see the original official README from NVIDIA [here](https://github.com/NVlabs/stylegan2-ada-pytorch) for details on licenses and citations.

## Compatibility
Note: This version of StyleGAN2 is not compatible with PyTorch>1.8. I use PyTorch 1.7 for my experiments.

## Datasets

Please see links for the datasets I used for my experiments - 

* TokWotel (Wood and Metal hits separated) - https://drive.google.com/file/d/1xjU868UgJBwnkrFEXlJg1S5u-SUNK6ag/view?usp=sharing
* A subset and pre-processed Greatest Hits Dataset - https://drive.google.com/file/d/1U3QRj3GQTlCcLj4BriSaWd3JYIP5sE4W/view?usp=sharing 

Please use the notebook called [pghi-test.ipynb](pghi-test.ipynb) to visualise the spectrogram representations.

## Training new networks

To training new networks use the commands below. Note that the datasets directory should contain the '*.wav' files with no sub-directory structure. Also, all my experiments were unconditional training. For conditioned training you will need an additional ```dataset.json``` as explained in the original NVIDIA README.

The flag --aug=noaug is important. The augmentations (rotation etc.,) used in the computer vision domain will not work for audio spectrograms learning.

```.bash
python train.py --outdir=training-runs --data=datasets/tokwotel --gpus=1 --aug=noaug --dry-run
python train.py --outdir=training-runs --data=datasets/tokwotel --gpus=1 --aug=noaug

python train.py --outdir=training-runs --data=datasets/vis-data-256-split --gpus=1 --aug=noaug --dry-run
python train.py --outdir=training-runs --data=datasets/vis-data-256-split --gpus=1 --aug=noaug
```

## Generate

We use PGHI method to generate Spectrograms. StyleGAN architectures for audio learn spectrogram representations as images and thus need to be scaled from [-50,0] to [0,255].
For this, please use the [generate-rescaled-final.ipynb](generate-rescaled-final.ipynb)
