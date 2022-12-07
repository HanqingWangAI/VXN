# VXN
-----------

<p align="left">
    <a href='https://arxiv.org/abs/2210.16822'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
</p>
<div align=center>
<img src='assets/overview.png' width=95%>
</div>

The official repository of the **NeurIPS 2022** paper: **Towards Versatile Embodied Navigation**. VXN is the abbreviation of *Vision-$X$ Navigation*, a large-scale test bed for multi-task embodied navigation.

[Hanqing Wang](https://hanqingwangai.github.io) | [Wei Liang](https://liangwei-bit.github.io/web/) | [Luc Van Gool](https://ee.ethz.ch/de/departement/professoren/professoren-kontaktdetails/person-detail.OTAyMzM=.TGlzdC80MTEsMTA1ODA0MjU5.html) | [Wenguan Wang](https://sites.google.com/view/wenguanwang/home?authuser=0)





## Environment Installation
-------
Create a python environment and install the required packages using the following scripts:
```bash
conda create -n vxn --file requirements.txt
conda activate vxn
```


## Dataset 
------
Create the folder for datasets using the following scripts:
```bash
mkdir data
```
### Matterport3D Scenes
Download the matterport3D dataset following the instruction [here](https://github.com/jacobkrantz/VLN-CE#scenes-matterport3d).

### Navigation Tasks
Download [datasets]() for `image-goal` nav., `audio-goal` nav., `object-goal` nav., and `vision-language` nav. tasks, and uncompressed it under the path `data/datasets`.

### Continuous Audio Rendering
1. Download the rendered BRIRs (binaural room impulse responses) (887G) for Matterport3D scenes [here](). Put `data.mdb` under the path `data/binaural_rirs_lmdb/`.
2. Download the alignment data (505G) for discrete BRIRs [here](). Put `data.mdb` under the path `data/align_lmdb/`.


## Training
---
For a multi-node cluster, run the following script to start the training.
```bash
bash sbatch_scripts/sub_job.sh
```

## Evaluation
---
Run the following script to evaluate the trained model for each task.
```bash
bash sbatch_scripts/eval_mt.sh
```
## TODOs
1. [ ] Release the full dataset.
2. [ ] Release the checkpoints.



## Citation
----
If you find our project useful, please consider citing us:

```bibtex
@inproceedings{wang2022towards,
  title = {Towards Versatile Embodied Navigation},
  author = {Wang, Hanqing and Liang, Wei and Van Gool, Luc and Wang, Wenguan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2022}
}
```

## License

The VXN codebase is [MIT licensed](LICENSE). Trained models and task datasets are considered data derived from the mp3d scene dataset. Matterport3D based task datasets and trained models are distributed with [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).


## Acknowledgment
This repository is built upon the following publicly released projects:
- [habitat-lab](https://github.com/facebookresearch/habitat-lab)
- [sound-spaces](https://github.com/facebookresearch/sound-spaces)
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE)

Thanks to the authors who create those great prior works.