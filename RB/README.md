***Folders and annotations***
- datasets
    - contains the annotations for RB in VIA and COCO format
- zenodo
    - contains the original annotations uploaded on [Zenodo](https://zenodo.org/record/3706863)
    - images need to be downloaded manually
- RB.json
    - contains the tags for RB done using letagger
    
***Notebooks***
- datasets.ipynb
    - exports RB in COCO and VIA format
- exploration_RB.ipynb
    - statistical exploration of RB
- prepare_data.ipynb
    - crops images from the Zenodo dataset to be browsed in letagger
    - generates a file containing the metadata of each image (height and width)
    - generates a VIA file containing each stock tables
- preprocessing.ipynb
    - preprocess the tags obtained through letagger for further processing
    - should be ran before exploration_RB.ipynb and datasets.ipynb
- visualization_example.ipynb
    - provides an example to visualize the original annotations of RB


