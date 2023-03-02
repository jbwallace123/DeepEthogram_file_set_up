%% Create dirs and move files for deepethogram

%rootdir = 'C:\Users\berri\OneDrive\Desktop\TEST'; 
function dir_maker(rootdir)

%create parent directory
new_dir = dir(fullfile(rootdir, '*.avi'));

for i = 1:length(new_dir)
    baseFileName = new_dir(i).name;
    new_path = char(baseFileName(1:end-4));
    mkdir(rootdir, new_path);
end 
%list csv and avi dirs
move_csv = dir(fullfile(rootdir, '*.csv'));
move_avi = dir(fullfile(rootdir, '*.avi'));

%move csvs with matching str
for ii = 1:length(new_dir)
    baseFolderName = new_dir(ii).name;
    baseFolderName = char(baseFolderName(1:end-4));
    csvName = move_csv(ii).name;
    csvPathFile = char(csvName);
    if startsWith(csvPathFile, baseFolderName)
        movefile(fullfile(rootdir, csvPathFile), fullfile(rootdir, baseFolderName))
    end
end
%move avi with matching str
for ii = 1:length(new_dir)
    baseFolderName = new_dir(ii).name;
    baseFolderName = char(baseFolderName(1:end-4));
    aviName = move_avi(ii).name;
    aviPathFile = char(aviName);
    if startsWith(aviPathFile, baseFolderName)
        movefile(fullfile(rootdir, aviPathFile), fullfile(rootdir, baseFolderName))
    end
end

end 
%dir_maker

    