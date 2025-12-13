# UseGeo Dataset Forum Discussion

## Initial Post - Dataset Availability

**Professor (Navid Dadkhah Tehrani)**

Hello All,

I put the main part of the UseGeo dataset that you need for training for your project inside this zip file in OneDrive:

[UseGeo.zip](https://wpi0-my.sharepoint.com/:u:/g/personal/ndadkhahtehrani_wpi_edu/EZVPSUIGd1tHlNt_58XiVuQBJE6uMgzMVDfYwNyr17E4nQLinks)

For the rest of the data, such as camera trajectory information, etc, you can directly get it from their dataset website by right-clicking and hitting download:
https://github.com/3DOM-FBK/usegeo

---

## Thread 1: Access Issues

**Tamar Boone** - Dec 5 2:14pm

Hello Professor,

When trying to download the zip file I got a message that said this:
[Permission required error]

Are you the one that has to approve it? If not, then I'm not sure how to get to the files.

-Tamar

### Reply from Professor - Dec 5 3:04pm

please try again. not sure why I asked for permission. I gave permission to all wpi emails.

https://wpi0-my.sharepoint.com/:u:/g/personal/ndadkhahtehrani_wpi_edu/EZVPSUIGd1tHlNt_58XiVuQBJE6uMgzMVDfYwNyr17E4nQ

---

## Thread 2: Download Script & Depth Map Issues

**Everett Wenzlaff** - Dec 6 6:52pm

A few things: I actually think I have a working script for downloading all the files from the UseGeo dataset website (see attached). It's janky and un-optimized, but it's worked for me. The website where the data is located is so unconventional that ChatGPT and I had a tough time figuring it out, and the only solution we could come up with was having it automate the clicking, scrolling, and downloading of individual files. Seems like all the downloads are generated dynamically (on demand) and the underlying HTML grid with all the file data is only loaded for files that are "in view". 

Independent from this the next thing is: I'm trying to understand the data within this dataset and figure out what/how to map to some sort of ground truth for the depth. I'm assuming the ground truth is located in the "depth maps" folders for each dataset (Dataset_#/Depth_resized/depth_maps), but all the .tiff files that I'm seeing are the same exact size and completely white when I look at them in my default photo viewer app. Is this normal? Or should I be using specific software to inspect the files?

I don't have this issue when I look at the .tiff or .jpeg files in the "undistorted images" folder

[Attachment: download_stuff_v13.py]

### Reply from Everett - Dec 6 11:29pm

I learned that I need to use something like Gimp to "stretch the contrast" of the tiffs. Supposedly a lot of this geographic data is captured in 32 bit floats where most standard images use 8 bit floats. So I'm able to get stuff that looks like this out of the white images:
[Shows successfully visualized depth map]

### Reply from Professor - Dec 7 11:29am

that's correct. you can't see the depth .tiff with regular image viewer.

yes we're training based off of depth resized folder which is the resized version of the original images.

---

## Thread 3: Dataset Size Concerns

**Everett Wenzlaff** - Dec 9 4:40pm

It's finally sinking in just how small the 'UseGeo' dataset is in comparison to the MidAir dataset. I understand that we're supposed to perform 'training' on this dataset with the pretrained weights, but there are only 3 trajectories. Wouldn't training on this dataset cause overfitting to some degree? Also, would the train/val split essentially have to be 2 trajectories/1 trajectory? Or for the context of this project, would it make more sense to pass the UseGeo dataset as an input to the model in 'evaluation' mode to serve as a test dataset instead? Unless I'm missing something, it just seems weird to train with such a small dataset

### Reply from Professor - Dec 9 8:08pm

It is not easy to find real flight datasets with IMU/GPS and camera.

You first evaluate how the pre-trained model perform for the UseGoe dataset. and then you re-train it on the UseGeo dataset.

You don't have to use the entire trajectory for training. for example, you can keep 20 percent of each trajectory for validation. Also your training only need two sequence of images and transformation between them, so you don't really have to think of training as trajectory-based training.

---

## Thread 4: Technical Issues with TensorFlow

**Paul Raynes** - Dec 9 10:45pm

Has anyone had success actually reading the .tiff files as a part of the dataloader? Because TensorFlow doesn't have a built in .decode_tiff method, I have been trying to get the one from TensorFlow-io to work, but it doesn't seem to like the fact that they are 32-bit. I get an error when I try to run: "memory: Sorry, can not handle images with 32-bit samples." When I try to use some 3rd party library to parse the files (i.e. OpenCV), I get stuck because the file names as passed in to the _decode_samples method are stored as Tensors, not raw strings, and because it is wrapped in an @tf.function, the tensors are in graph mode and dont have the ususal .numpy() method on them to extract the raw string.

---

## Key Takeaways

1. **Dataset Location**: Main data in OneDrive, additional camera trajectory data on GitHub
2. **Depth Maps**: Located in `Dataset_#/Depth_resized/depth_maps` folders
3. **File Format Issue**: Depth .tiff files use 32-bit floats (not standard 8-bit), appear white in normal viewers
4. **Visualization**: Need tools like Gimp to "stretch contrast" to view depth maps properly
5. **Dataset Size**: Only 3 trajectories available (real flight data with IMU/GPS/camera is rare)
6. **Training Approach**: 
   - First evaluate pretrained model on UseGeo
   - Then retrain on UseGeo dataset
   - Can split each trajectory (e.g., 80% train / 20% validation)
   - Training is image-pair based, not trajectory-based
7. **Technical Challenge**: TensorFlow struggles with 32-bit .tiff files in dataloader