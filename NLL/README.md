***Folder and annotations***
- datasets
    - contains the splits for NLL, NLL-filtered and NLL-revised in VIA and COCO format, as well as NLL-auto and NLL-tag in a custom format in Parquet
- inconsistencies.parquet
    - contains the annotations made during the analysis of the inconsistencies in NLL
- NLL_auto.json
    - contains the tags for NLL-auto done using letagger with automatic augmentation 
- NLL_metadata_images.json
    - contains the metadata of each image (height, width) for each image in NLL
    - contains the resized height and width used for the resized images stored (to reduce total size since images are resized when fed to models)
- NLL_tag.json
    - contains the tags for NLL-tag done using letagger
- via_annotations_NLL_revised.json
    - contains the annotations of NLL-revised in VIA format

***Notebooks***
- datasets_table_classification.ipynb
    - exports the splits of NLL-tag and NLL-auto in a custom format in Parquet for the task of table classification
- datasets_table_detection.ipynb
    - exports the splits of NLL, NLL-filtered, NLL-revised and NLL-tag in COCO and VIA format for the task of table detection
- exploration_NLL_auto.ipynb
    - statistical exploration of NLL-auto
- exploration_NLL_tag.ipynb
    - statistical exploration of NLL-tag
- exploration_NLL.ipynb
    - statistical exploration of NLL
- filtering_inconsistencies.ipynb
    - contains the visualizations and analysis done on the inconsistencies of NLL
- prepare_data.ipynb
    - queries the annotations for tables on SOLR and IML
    - generates a file containing all the visual and textual informations of these tables
    - creates many additional files necessary for impresso-images tools, to download images from the IIIF of the NLL, store data on impresso's S3, to store visual signatures for each tables on S3 and to indicate image location for letagger
    - should be ran first
    - necessary to have access to impresso's SOLR and IML
- preprocessing.ipynb
    - preprocess the data obtained through prepare_data.ipynb for further processing and exploration
- visualization_example.ipynb
    - provides an example to visualize the original annotations of NLL