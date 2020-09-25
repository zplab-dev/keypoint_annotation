# Human reproducibility tests for keypoint annotations

Thank you for engaging in this incredibly important scientific endeavor! You are making a difference in advancing (my) science. I greatly appreciate your sacrifice.

## Motivation/why should you do this:

In this module you will be annotating the anatomical landmarks for adult C. elegans. These landmarks are “anterior pharyngeal bulb”, “posterior pharyngeal bulb”, “vulva”, and “tail” (examples in other sections). I need human annotated keypoints to show how variable human-annotated values are vs the neural net. Since I am a biased individual and am too close to the data, I need your help in generating this human dataset! 

###  Other reasons you should do this:
  1.	It will greatly help me
  2.	I’m pretty awesome so you should help me
  3.	You’re pretty awesome and care about science
  4.	It shouldn’t take too long

## Instructions:
I have written up a script that will open a riswidget annotator window with the 200 adult C. elegans images you need to annotate. All data and scripts are stored on Lugia.

###  Before you run the script:
1.	Ensure you have the zpl environments set up. If you are just starting out refer to [this Github post](https://github.com/zplab/protocols/blob/master/computer%20protocols/zplab%20Python%20Environment.md) on how to set up your environment.
    - If you have not recently updated elegant or ris_widget, please update it using (change elegant for specific packages): 
    ```pip install -U git+https://github.com/zplab/elegant```
2.	Connect to the VPN
    - see [this link](https://it.wustl.edu/items/connect/) for info on how to connect to the wustl VPN
3.  Connect to Lugia
    ##### For Mac:
    - Open Finder and click "Go" in the toolbar
    - Click "Connect to Server..."
    - Type 'smb://lugia.wucon.wustl.edu' into the Server Address bar and click Connect
    - Connect to lugia_array using the zplab login
    ##### For Ubuntu:
    - In the file manager click "Other Locations" in the sidebar
    - In "Connect to Server", enter 'smb://lugia.wucon.wustl.edu' and click Connect
    >(Note that I haven't tried this way since I don't have an Ubuntu computer on hand. Please let me know if there's a different way to do this with an Ubuntu computer)
4.  Make sure the script is in a place you can run locally
    - Option 1: Download the 'annotate_experiment_script.py' from this folder
    - Option 2: Clone the whole keypoint_annotations repository ([Click here](https://docs.github.com/en/enterprise/2.13/user/articles/cloning-a-repository) for info on how to clone a Github repository)
    
### Running the script:

### Annotating the worms:
#### Examples:

## Troubleshooting problems:

