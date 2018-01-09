clear;clc;close all;

%% configurations
folder='C:\Users\Sheshan\Dropbox\HKU_notes\ELEC_6081_Biomedical_Signals_and_Systems\assignment3\assignment3\assignment3'; % folder of images

marg=10; % desired margin

%% get image files
files=dir([folder '/*.png']);
files=[files; dir([folder '/*.PNG'])];
files=[files; dir([folder '/*.jpg'])];
files=[files; dir([folder '/*.JPG'])];
files=[files; dir([folder '/*.bmp'])];
files=[files; dir([folder '/*.BMP'])];

%% remove white margin
for i=1:length(files)
    filepath=[folder '/' files(i).name];
    im=imread(filepath);
    im=double(im);
    
    up=1;
    down=size(im,1);
    left=1;
    right=size(im,2);
    
    while sum(sum(255-im(up,:,:)))==0
        up=up+1;
    end
    
    while sum(sum(255-im(down,:,:)))==0
        down=down-1;
    end
    
    while sum(sum(255-im(:,left,:)))==0
        left=left+1;
    end
    
    while sum(sum(255-im(:,right,:)))==0
        right=right-1;
    end
    
    if up-marg>=1
        up=up-marg;
    end
    
    if down+marg<=size(im,1)
        down=down+marg;
    end
    
    if left-marg>=1
        left=left-marg;
    end
    
    if right+marg<=size(im,2)
        right=right+marg;
    end
    
    im=im(up:down,left:right,:);
    
    im=uint8(im);
    imwrite(im,filepath);
end

