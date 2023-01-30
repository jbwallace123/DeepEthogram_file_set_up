%% readCSV

csv_path ='C:\Users\berri\OneDrive\Desktop\TEST';
myFiles = dir(fullfile(csv_path,'*.csv'));


for i=1:length(myFiles)
    baseFileName = myFiles(i).name;
    fullFileName = fullfile(csv_path, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    
    csv = readtable(fullFileName, 'PreserveVariableNames',true); 

    csv = renamevars(csv, ["Var1", "Var2", "Var3", "Var4", "Var5", "Var6"],...
        ["Frame", "success", "drop", "miss", "missing_pellet", "background"]);

    csv_new = [csv(:,1) csv(:,6) csv(:,2:5)];

    save_fname = char(fullFileName);
    writetable(csv_new, save_fname);

    disp('File Saved');
end

    