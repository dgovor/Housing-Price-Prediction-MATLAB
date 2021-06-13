% The function's input: cell array; word that needs to be replaced; number
% that we are replacing the word with; column that is a subject to changes
% Example: data = replace_func(data,'Inside',2,10);
% All "Inside" words in column #10 of the cell array "data" will be changed to number 2

function M = replace_func(initial_array,word,number,feature)

M = initial_array; % Saving our array

for i = 1:size(M,1) % for loop goes through each data sample of the picked feature
    a = M{i,feature}; % Each word is saved
    if length(a) == length(word) % Check if the length of the picked word corresponds to the length of desired word
        if a(1:length(word)) == word % Checking if the words are the same
            M{i,feature} = number; % Changing the pciked word for the desired number
        end
    end 
end