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
    * If you have not recently updated elegant or ris_widget, please update it using (change elegant for specific packages): 
    
       ```pip install -U git+https://github.com/zplab/elegant```
2.	Connect to the VPN: see [this link](https://it.wustl.edu/items/connect/) for info on how to connect to the wustl VPN
3.  Connect to Lugia
    ##### For Mac:
    * Open Finder and click "Go" in the toolbar
    * Click "Connect to Server..."
    * Type 'smb://lugia.wucon.wustl.edu' into the Server Address bar and click Connect
    * Connect to lugia_array using the zplab login
    ##### For Ubuntu:
    * In the file manager click "Other Locations" in the sidebar
    * In "Connect to Server", enter 'smb://lugia.wucon.wustl.edu' and click Connect
    
    (Note that I haven't tried this way since I don't have an Ubuntu computer on hand. Please let me know if there's a different way to do this with an Ubuntu computer)
4.  Make sure the script is in a place you can run locally
    * Option 1: Download the 'annotate_experiment_script.py' from this folder
    * Option 2: Clone the whole keypoint_annotations repository ([Click here](https://docs.github.com/en/enterprise/2.13/user/articles/cloning-a-repository) for info on how to clone a Github repository)
    
### Running the script:
1.  Open a terminal window and navigate to where you saved the annotation script
2.  Run the script with your initials in the argument slot (my nitials are NML, so replace that bit with your initials):
    
    ```python annotate_experiment_script.py NML```
3.  A RisWidget window with a keypoint annotator and pose annotator should come up. Note that it might take a little while for all the images to load over the network, so go get a snack while you wait!
4.  Annotate the worms!
### Annotating the worms:
1.  The keypoints you will be annotating are (see Examples for visual references):
    * Anterior pharyngeal bulb: posterior end of the anterior pharyngeal bulb 
    * Posterior pharyngeal bulb: posterior end of the posterior pharyngeal bulb
    * Vulva: middle of the vulva 
    * Tail: end of visible tissue/texture (where the worm tissue goes from more textured to clear) in the posterior end of the worm
2.  Click the 'Lock' check box in the annotator
3.  Click the locations of each keypoint in the **straightened worm image**
    * Keypoint clicks will be placed in the order "Anterior bulb", "Posterior bulb", "vulva", "tail" and will be color coordinated
    * You will be able to see which keypoints have been added in the bottom of the annotator (colored keypoint names indicate the keypoint has been placed)

> The straightened worm image should be situated with the head to the left and the tail to the right. If this is **not** the case, click the 'reverse' button to reverse the direction the worm is facing in

4. You can hit 'Save' or otherwise just exit out of the RisWidget window to save your annotations. 
    
#### Examples:
1.  Annotator example:

![Annotator Example](https://github.com/zplab-dev/keypoint_annotation/blob/master/keypoint_annotation/reproducibility/Examples/annotator_ex.png)

2.  Annotated worm examples: Colored points correspond with the keypoints

![Annotated Worm Examples: Color points correspond with the keypoints.](https://github.com/zplab-dev/keypoint_annotation/blob/master/keypoint_annotation/reproducibility/Examples/examples_legend.png)

![Additional Annotated worm examples](https://github.com/zplab-dev/keypoint_annotation/blob/master/keypoint_annotation/reproducibility/Examples/examples1.png)

## Tips and tricks for annotating

1.  Use the histogram brightness/contrast toggle.
    * Alter the brightness/contrast to see the keypoints easier.
2.  Try your best! Some of these worms are really old and tricky to find the keypoints on.
3.  Take breaks! Staring at worms can make your eyes tired. If you feel like you're seeing shapes in the clouds, take a break and come back to it later.
4.  You can always ask me if you have questions!

### Tips for annotating specific keypoints:
1.  Anterior Bulb: Look for circles
    * The pharynx is basically 2 circles, so look for the the most anterior circle.
    * Dotted lines indicate where I've identified the "circles" of the anterior pharyngeal bulb
    
    ![Tips and tricks for anterior bulbs](https://github.com/zplab-dev/keypoint_annotation/blob/master/keypoint_annotation/reproducibility/Examples/Anterior_bulb_tips1.gif)

2.  Posterior Bulb: Look for circles posterior of the anterior bulb
    * The Posterior bulb is the last stop before the intestine. If you can find the intestine try to see if you can find a circlular thing attached to it.
    * Dotted lines indicate where I've identified the "circles" of the posterior pharyngeal bulb
    
    ![Tips and tricks for posterior bulbs](https://github.com/zplab-dev/keypoint_annotation/blob/master/keypoint_annotation/reproducibility/Examples/posterior_bulb_tips.gif)
    
3.  Vulva: Look for protrusions, "channels", and light spots
    * The vulva is generally near the middle of the worm. You can sometimes see the vulva protruding out the side of the worm.
    * You can often see the opening of the vulva that looks like a "channel" or the coming together of two parts of the worm.
    * The vulva will often be at the intersection of two light spots in the middle of the worm, where you can also see the "channel"
    
    ![Tips and tricks for vulva](https://github.com/zplab-dev/keypoint_annotation/blob/master/keypoint_annotation/reproducibility/Examples/Vulva_tips.png)
    
    ![Tips and tricks for vulva ex2](https://github.com/zplab-dev/keypoint_annotation/blob/master/keypoint_annotation/reproducibility/Examples/vulva_tips1.png)
    
7.  Tail: Look for texture changes at the end of the worm
    * Texture changes at the end of the worm will often go from "dark" to "light". The intestine tapers off towards the end of the worm. Look for that.
    * This is the most arbitrary point, so just do your best.
    
    ![Tips and tricks for tail](https://github.com/zplab-dev/keypoint_annotation/blob/master/keypoint_annotation/reproducibility/Examples/tail_tips.png)
    
Any questions/comments can be addressed to me!
Thank you again so much for your help!!!
