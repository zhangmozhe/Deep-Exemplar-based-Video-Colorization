# Deep Exemplar-based Video Colorization (Pytorch Implementation)

<img src='assets/teaser.png'/>

### [Paper](https://arxiv.org/abs/1906.09909) | [Pretrained Model]() | [Youtube video](https://youtu.be/HXWR5h5vVYI)

**Deep Exemplar-based Video Colorization, CVPR2019**

[Bo Zhang](https://www.microsoft.com/en-us/research/people/zhanbo/)<sup>1,3</sup>,
[Mingming He](http://mingminghe.com/)<sup>1,5</sup>,
[Jing Liao](https://liaojing.github.io/html/)<sup>2</sup>,
[Pedro V. Sander](https://www.cse.ust.hk/~psander/)<sup>1</sup>,
[Lu Yuan](https://www.microsoft.com/en-us/research/people/luyuan/)<sup>4</sup>,
[Amine Bermak](https://eebermak.home.ece.ust.hk/)<sup>1</sup>,
[Dong Chen](https://www.microsoft.com/en-us/research/people/doch/)<sup>3</sup> <br>
<sup>1</sup>Hong Kong University of Science and Technology,<sup>2</sup>City University of Hong Kong,
<sup>3</sup>Microsoft Research Asia, <sup>4</sup>Microsoft Cloud&AI, <sup>5</sup>USC Institute for Creative Technologies

## Prerequisites

- Python 3.6+
- Nvidia GPU + CUDA, CuDNN

## Installation

```bash
conda create -n ColorVid python=3.6
source activate ColorVid
pip install -r requirements.txt
```

## Data preparation

checkpoint, pre-trained vgg

- input video frames
- reference image
- output

## Test

```bash
python test.py
```

## TODO

- [ ] Release the training code
- [ ] Prepare the Colab Demo
- [ ] Output the video
- [ ] GIF teaser

## Citation

If you use this code for your research, please cite our paper.

> @inproceedings{zhang2019deep,
> title={Deep exemplar-based video colorization},
> author={Zhang, Bo and He, Mingming and Liao, Jing and Sander, Pedro V and Yuan, Lu and Bermak, Amine and Chen, Dong},
> booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
> pages={8052--8061},
> year={2019}
> }

## License

This project is licensed under the MIT license.
