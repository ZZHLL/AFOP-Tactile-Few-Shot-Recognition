function [files,labels] = cwt_files_from_rows(imageRoot,features,rows)
% Map feature-table rows to class/trial CWT image files.

rows=rows(:);
labels=features.y_class(rows);labels=labels(:);
trials=features.y_trial(rows);trials=trials(:);
files=strings(numel(rows),1);
for i=1:numel(rows)
    globalNumber=(labels(i)-1)*60+trials(i);
    files(i)=fullfile(imageRoot,sprintf('S%d',labels(i)),sprintf('%d.jpg',globalNumber));
end
end
