function fill = fill_batch(myDir)
%% introductory stuff
cd(myDir);
if exist('dict.txt', 'file')==2
    temp=importdata('dict.txt');
    temp=temp{1};
    dict=strsplit(temp,',');
end
load batch.mat;
matlabbatch{1, 1}.spm.spatial.realign.estimate.data=[];
need=myDir;
path=cd('../');
currentDirectory = pwd;
[upperPath, working, ~] = fileparts(currentDirectory); 
working=lower(working);
cd(path);
%% filling data for realign
cd('epi');
a=dir('*nii');
size=length(a);
fid=fopen('../dict_entries.txt','w');
for i=1:size;
    name=a(i).name;
    v=spm_vol(name);
    frames=length(v);
    if(exist('dict'))==1;
        for j=1:length(dict);
           if isnan(strfind(name,dict{j}))==0;
               inpnames='Input';
               numlines=1;
               defaultanswer={''};
               options.WindowStyle='normal';
               spm_check_registration(name);
               inp = inputdlg(strcat('How many relevant frames are in',{' '},name,' ?'),inpnames,numlines,defaultanswer,options);
               inp=inp{1};
               ent=str2num(inp);
               while(ent>frames);
                   inpnames='Input';
                   numlines=1;
                   defaultanswer={''};
                   options.WindowStyle='normal';
                   inp = inputdlg('Selected more relevant frames than frames in file, please try again..',inpnames,numlines,defaultanswer,options);;
                   inp=inp{1};
                   ent=str2num(inp);
               end;
               fprintf(fid,'%s has %d relevant frames\n', name, ent);
               break
           end;
        end;
        
    end;
    close gcf;
    if(exist('ent'))==1
        field=cell(ent,1);
        for j=1:ent;
            field{j,1}=strcat(need,'/epi/',name,',',num2str(j));
        end;
    else
        field=cell(frames,1);
        for j=1:frames;
            field{j,1}=strcat(need,'/epi/',name,',',num2str(j));
        end
    end
            
    matlabbatch{1, 1}.spm.spatial.realign.estimate.data{1,i}=field;
    clearvars ent inp
end
fclose(fid);
cd('../')
%% fill in hires
cd('hires');
highres=dir('*mprage*.nii');
if length(highres)==1;
    hires=highres.name;
else
    shortestNum=length(highres(1).name);
    shortestName=highres(1).name;
    for i=1:length(highres);
        cur=length(highres(i).name);
        if cur<shortestNum
            shortestNum=length(highres(i).name);
            shortestName=highres(i).name;
        end
    end
    hires=shortestName;
end
matlabbatch{1, 2}.spm.spatial.coreg.estwrite.ref{1, 1}=strcat(need,'/hires/',hires,',1');
matlabbatch{1, 3}.spm.spatial.preproc.channel.vols{1, 1}=strcat(need,'/hires/',hires,',1');
cd('..');

%% fill in other images
load 'template.mat';
for i=1:size
    matlabbatch{1, 2}.spm.spatial.coreg.estwrite.other(1, i)=template;
    matlabbatch{1, 2}.spm.spatial.coreg.estwrite.other(1, i).src_output(2).subs{1, 1}=i;
    q=char(strcat('Realign: Estimate: Realigned Images (Sess ',{' '},num2str(i),')'));
    matlabbatch{1, 2}.spm.spatial.coreg.estwrite.other(1, i).sname=q;
end
matlabbatch{1, 2}.spm.spatial.coreg.estwrite.other=matlabbatch{1, 2}.spm.spatial.coreg.estwrite.other(1:size);
%% fill in tissue probability map input
if exist('tissue.txt','file')==2;
    tissue=textread('tissue.txt','%s');
    for i=1:6
        matlabbatch{1, 3}.spm.spatial.preproc.tissue(i).tpm=strcat(tissue,',',num2str(i));
    end
%% save matlabbatch
end
batch=matlabbatch;
save('batch');