function map=mapping(img)
%%prepoced row
Vepi = spm_vol(img);
Yepi = spm_read_vols(Vepi);
si=size(Yepi);
ax1=subplot(4,3,10);
b=squeeze(Yepi(round((si(1))/2),:,:)); %sagittal plane
b=imrotate(b,90);
imagesc(b);
colormap(ax1,gray);
title('Preprocessed Sagittal Plane')
set(gca,'XTickLabel','');
set(gca,'YTickLabel','');
ax2=subplot(4,3,11);
c=squeeze(Yepi(:,round((si(2))/2),:)); %coronal plane
c=imrotate(c,90);
imagesc(c);
colormap(ax2,gray);
title('Preprocessed Coronal Plane');
set(gca,'XTickLabel','');
set(gca,'YTickLabel','');
ax3=subplot(4,3,12);
d=squeeze(Yepi(:,:,round((si(3))/2))); %axial plane
imagesc(d);
colormap(ax3,gray);
title('Preprocessed Axial Plane');
set(gca,'XTickLabel','');
set(gca,'YTickLabel','');

%% template row
temp=which('spm_jobman');
spmPath=temp(1:end-12);
canon=strcat(spmPath,'canonical/');
template=dir(strcat(canon,'*152T1*'));
templateName=template.name;
templatePath=strcat(canon,templateName);
Vepi = spm_vol(templatePath);
Yepi = spm_read_vols(Vepi);
si=size(Yepi);
ax1=subplot(4,3,7);
b=squeeze(Yepi(round((si(1))/2),:,:)); %sagittal plane
b=imrotate(b,90);
imagesc(b);
colormap(ax1,gray);
title('Template Sagittal Plane')
set(gca,'XTickLabel','');
set(gca,'YTickLabel','');
ax2=subplot(4,3,8);
c=squeeze(Yepi(:,round((si(2))/2),:)); %coronal plane
c=imrotate(c,90);
imagesc(c);
colormap(ax2,gray)
title('Template Coronal Plane');
set(gca,'XTickLabel','');
set(gca,'YTickLabel','');
ax3=subplot(4,3,9);
d=squeeze(Yepi(:,:,round((si(3))/2))); %axial plane
imagesc(d);
colormap(ax3,gray);
title('Template Axial Plane');
set(gca,'XTickLabel','');
set(gca,'YTickLabel','');
end