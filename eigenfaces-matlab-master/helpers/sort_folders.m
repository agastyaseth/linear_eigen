function [res] = sort_folders(dir_listing)
    %sort_folders: sort folder using a natural sort.
    % usage: [res] = sort_folders(dir_listing)

    j = 1;
    for i = 1 : length(dir_listing)
        if dir_listing(i).isdir
            nat_sort_dir_listing(j,:)=cellstr(dir_listing(i).name);
            j = j + 1;
        end
    end
    res = sort_nat(nat_sort_dir_listing);
end