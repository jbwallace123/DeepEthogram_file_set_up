%% readCSV
%for new codec code that handles short videos and long for Kim's two
%different rigs

csv_path ='C:\Users\berri\OneDrive\Desktop\TEST';
myFiles = dir(fullfile(csv_path,'*.csv'));


for i=1:length(myFiles)
    baseFileName = myFiles(i).name;
    fullFileName = fullfile(csv_path, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    
    csv = readtable(fullFileName,'PreserveVariableNames',true); 
    
      if ~exist('Var6')
          Var6 = [];
          Var6 = zeros(length(csv.Var5),1);
          Var6(csv.Var2==0 & csv.Var3==0 & csv.Var4==0 & csv.Var5==0)=1;
      end
      
    csv.Var6 = Var6;           
    csv = renamevars(csv, ["Var1", "Var2", "Var3", "Var4", "Var5", "Var6"],...
        ["Frame", "success", "drop", "miss", "missing_pellet", "background"]);

    csv_new = [csv(:,1) csv(:,6) csv(:,2:5)];
    
    Name = char(fullFileName);
    Name = Name(1:end-4);
    save_fname = [Name '_' 'labels' '.csv'];
    writetable(csv_new, save_fname);

    disp('File Saved');
    end
 
   
        

    