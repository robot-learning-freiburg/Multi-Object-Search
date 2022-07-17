# Learning Long-Horizon Robot Exploration Strategies for Multi-Object Search in Continuous Action Spaces

Repository providing the source code for the paper "Learning Long-Horizon Robot Exploration Strategies for Multi-Object Search in Continuous Action Spaces", see the [project website](http://multi-object-search.cs.uni-freiburg.de). Please cite the paper as follows:

    @article{fabian22exploration,
	  title={Learning Long-Horizon Robot Exploration Strategies for Multi-Object Search in Continuous Action Spaces},
	  author={Schmalstieg, Fabian and Honerkamp, Daniel and Welschehold, Tim and Valada, Abhinav},
	  journal={Proceedings of the International Symposium on Robotics Research (ISRR)},
	  year={2022}
    }
  
    
# Installation

```
conda create -f environment.yaml
conda activate igibson
```

The code does not work without downloading and unzipping the iGibson dataset.
First download assets and then, the dataset of the scenes. 
Unzip the dataset into the iGibson/data/. folder. After the unzipping, the iGibson/data folder should have a ig_dataset folder.
Don't forget to download the igibson.key files under https://stanfordvl.github.io/iGibson/dataset.html
 
```
python -m igibson.utils.assets_utils --download_assets
https://storage.googleapis.com/gibson_scenes/ig_dataset.tar.gz
```

After Installing SB3 and iGibson, copy all files which are located in requirements/ , into the respective folders of iGibson/data. You have to overwrite the existing files. 
These files are only used during training in order to use inflated maps.


# Run Evaluation

run the model with different scenes using 

```

python evaluate.py

```

change the scene_id name in config.yaml to one of the following scenes:

Test Scenes:
Benevolence_1_int, Wainscott_0_int, Pomaria_2_int, Benevolence_2_int, Beechwood_0_int, Pomaria_1_int, Merom_1_int

Training Scenes:
Merom_0_int, Benevolence_0_int, Pomaria_0_int, Wainscott_1_int, Rs_int, Ihlen_0_int, Beechwood_1_int, Ihlen_1_int

# Run Training

run training using:

```

python training.py

```


# Acknowledgements

This work was funded by the european union's horizon 2020 research and innovation program under grant agreement no 871449-OpenDR.
