# DeepEthogram_file_set_up

convertLowSpeedAVIs.py takes video data acquired by DVR and replaces codec, it also seperates videos into smaller clips for easier handling along with seperating behavior labels by Kim's SVM. 
To run: 
cd FILEDIR
python /PATH/to/python/code/convertLowSpeedAVIs.py .

- - - -
reorderCSV.m takes csv labels and reorders them for deepethogram training. Need to edit path to csv file directory.
- - - -
dir_maker function is used to create the directory structure needed for deepethogram if you have too many files to use the embedded code within the deepethogram workflow to import videos/labels. You will need the AVI video and the associated _labels csv. Then, you can execute the fucntion by running dir_maker(YOUR/FILE/DIR)
