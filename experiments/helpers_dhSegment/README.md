- config_dhSegment.json
    - an example of a config file to be used when running an experiment using dhSegment
- predict_dhSegment.py
    - infer predictions with a dhSegment model on images
- prepare_dhSegment.py
    - prepare anything necessary to run an experiment using dhSegment
    - should be ran first
- results_dhSegment.py
    - after having generated predictions using predict_dhSegment.py, computes results in terms of mIoU
- train_dhSegment.py
    - trains dhSegment using information given in a config.json file following the format of config_dhSegment.json
- visualization_dhSegment.ipynb
    - provides an example to visualize the predictions of a dhSegment model