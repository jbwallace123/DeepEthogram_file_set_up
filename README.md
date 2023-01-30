# DeepEthogram_file_set_up

convertLowSpeedAVIs.py takes video data acquired by DVR and replaces codec, it also seperates videos into smaller clips for easier handling along with seperating behavior labels by Kim's SVM. 
cd FILEDIR
python /PATH/to/python/code/convertLowSpeedAVIs.py .

reorderCSV.m takes csv labels and reorders them for deepethogram training. Need to edit path to csv file directory.
