% Author: Kyla Bouldin
% Description: creates training data file based on arial letter characters
% (10x15 image)

% trainingData = []
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'];

for i=1:26
    filename = sprintf('trainingdata/%s.png', alphabet(i));
    letter = imread(filename);
end