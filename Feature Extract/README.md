This code contained in this folder is to extract the feature of .mp3 file via the soundnet. Here is the tensorflow version of soundnet, implemented by Hou-Ning Hu. To figure how to use this tensorflow version of soundnet, please refer to
[@eborboihuc](https://github.com/eborboihuc).

The data in `./data` folder is a small sample of demo among the whole data set. To use this data, download all the .mp3 file via google drive, and run **extract_feat.py** as instructed in [extract feature](https://github.com/eborboihuc/SoundNet-tensorflow)

The extracted feature is in ".npy" format, which will be used in **svm classification** step, which can be seen in the upper folder.

