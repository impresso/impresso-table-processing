letagger is a Flask based application using the Bootstrap library for HTML and CSS. It can be used to tag cropped out images from larger images. The necessary files to make it work are generated through the notebooks prepare_data.ipynb. The code in settings.py file should be modified to fit the different paths to the data sources.

- static
    - contains jQuery
    - contains the CSS file of the GUI
    - contains the JS file that handle all the GUI front-end logic
- templates 
    - contains the HTML file for the GUI
- cache.py
    - handles the logic of the cached data of the application
- helpers
    - contains helper methods
- main.py
    - the main Flask file which handles the API of the web application
- settings.py
    - contains all the metadata on the dataset to be annotated