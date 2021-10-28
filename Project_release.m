clc
clear
close all

% Forcing readtable to read all the values from the datasets as "char"
% values. Otherwise, readtable can work differently in different versions
% of MATLAB and on different OS
opts1 = detectImportOptions('train.csv','PreserveVariableNames',true);
opts1 = setvartype(opts1, 'char');
opts2 = detectImportOptions('test.csv','PreserveVariableNames',true);
opts2 = setvartype(opts2, 'char');
% Reading input files as tables
train_ref = readtable('train.csv',opts1);
test_ref = readtable('test.csv',opts2);

% Converting tables into cell arrays deleting the headers
train = table2cell(train_ref);
test = table2cell(test_ref);

% Combining train and test data sets so it would be easier to clean data
data_set = [train; test num2cell(zeros(size(test,1),1))];

% Finding each feature that contains missing data
[~,j] = find(strcmp(train,'NA'));
% Calculating the number of missing values in each feature
[g_count,g_index] = groupcounts(j);
index = 0;

% For each feature that contains missig values we calculate percentage of
% missing values. If it is higher than 50% we delete this feature since the
% majority of data is missing
for i = 1:length(g_index)
    if g_count(i)/1460*100 > 50 % Check if the percentage is higher than 50%
        data_set(:,g_index(i) - index) = []; % Delete the feature with missing data
        index = index + 1;
    end
end

% Next 300 lines we encode all categorical fetures into numeric features.
% First, numeric values were assigned in numerical order. Later we change
% these values so it would be possible to describe them linearly

% The encoder was written in separate function
% The function's input: cell array; word that needs to be replaced; number
% that we are replacing the word with; column that is a subject of changes

data_set = replace_func(data_set,'A',6,3);
data_set = replace_func(data_set,'C (all)',1,3);
data_set = replace_func(data_set,'FV',5,3);
data_set = replace_func(data_set,'I',7,3);
data_set = replace_func(data_set,'RH',3,3);
data_set = replace_func(data_set,'RL',4,3);
data_set = replace_func(data_set,'RP',8,3);
data_set = replace_func(data_set,'RM',2,3);

% We delete "Street" because of its low variance.

% data_set = replace_func(data_set,'Grvl',1,6);
% data_set = replace_func(data_set,'Pave',2,6);

data_set = replace_func(data_set,'Reg',1,7);
data_set = replace_func(data_set,'IR1',2,7);
data_set = replace_func(data_set,'IR2',4,7);
data_set = replace_func(data_set,'IR3',3,7);

data_set = replace_func(data_set,'Lvl',2,8);
data_set = replace_func(data_set,'Bnk',1,8);
data_set = replace_func(data_set,'HLS',4,8);
data_set = replace_func(data_set,'Low',3,8);

% We delete "Utilities" because of its extremely low variance. All data
% samples have the same value except for the one data sample.

% data_set = replace_func(data_set,'AllPub',1,9);
% data_set = replace_func(data_set,'NoSewr',2,9);
% data_set = replace_func(data_set,'NoSeWa',3,9);
% data_set = replace_func(data_set,'ELO',4,9);

data_set = replace_func(data_set,'Inside',2,10);
data_set = replace_func(data_set,'Corner',3,10);
data_set = replace_func(data_set,'CulDSac',5,10);
data_set = replace_func(data_set,'FR2',1,10);
data_set = replace_func(data_set,'FR3',4,10);

data_set = replace_func(data_set,'Gtl',2,11);
data_set = replace_func(data_set,'Mod',3,11);
data_set = replace_func(data_set,'Sev',1,11);

data_set = replace_func(data_set,'Blmngtn',16,12);
data_set = replace_func(data_set,'Blueste',8,12);
data_set = replace_func(data_set,'BrDale',3,12);
data_set = replace_func(data_set,'BrkSide',4,12);
data_set = replace_func(data_set,'ClearCr',19,12);
data_set = replace_func(data_set,'CollgCr',17,12);
data_set = replace_func(data_set,'Crawfor',18,12);
data_set = replace_func(data_set,'Edwards',5,12);
data_set = replace_func(data_set,'Gilbert',15,12);
data_set = replace_func(data_set,'IDOTRR',2,12);
data_set = replace_func(data_set,'MeadowV',1,12);
data_set = replace_func(data_set,'Mitchel',12,12);
data_set = replace_func(data_set,'NAmes',11,12);
data_set = replace_func(data_set,'NoRidge',25,12);
data_set = replace_func(data_set,'NPkVill',10,12);
data_set = replace_func(data_set,'NridgHt',24,12);
data_set = replace_func(data_set,'NWAmes',14,12);
data_set = replace_func(data_set,'OldTown',6,12);
data_set = replace_func(data_set,'SWISU',9,12);
data_set = replace_func(data_set,'Sawyer',7,12);
data_set = replace_func(data_set,'SawyerW',13,12);
data_set = replace_func(data_set,'Somerst',20,12);
data_set = replace_func(data_set,'StoneBr',23,12);
data_set = replace_func(data_set,'Timber',22,12);
data_set = replace_func(data_set,'Veenker',21,12);

data_set = replace_func(data_set,'Artery',1,13);
data_set = replace_func(data_set,'Feedr',3,13);
data_set = replace_func(data_set,'Norm',4,13);
data_set = replace_func(data_set,'RRNn',7,13);
data_set = replace_func(data_set,'RRAn',5,13);
data_set = replace_func(data_set,'PosN',8,13);
data_set = replace_func(data_set,'PosA',9,13);
data_set = replace_func(data_set,'RRNe',6,13);
data_set = replace_func(data_set,'RRAe',2,13);

% We delete "Condition2" because of its low variance.

% data_set = replace_func(data_set,'Artery',1,14);
% data_set = replace_func(data_set,'Feedr',2,14);
% data_set = replace_func(data_set,'Norm',3,14);
% data_set = replace_func(data_set,'RRNn',4,14);
% data_set = replace_func(data_set,'RRAn',5,14);
% data_set = replace_func(data_set,'PosN',6,14);
% data_set = replace_func(data_set,'PosA',7,14);
% data_set = replace_func(data_set,'RRNe',8,14);
% data_set = replace_func(data_set,'RRAe',9,14);

data_set = replace_func(data_set,'1Fam',5,15);
data_set = replace_func(data_set,'2fmCon',1,15);
data_set = replace_func(data_set,'Duplex',2,15);
data_set = replace_func(data_set,'TwnhsE',4,15);
data_set = replace_func(data_set,'Twnhs',3,15);

data_set = replace_func(data_set,'1Story',6,16);
data_set = replace_func(data_set,'1.5Fin',3,16);
data_set = replace_func(data_set,'1.5Unf',1,16);
data_set = replace_func(data_set,'2Story',7,16);
data_set = replace_func(data_set,'2.5Fin',8,16);
data_set = replace_func(data_set,'2.5Unf',4,16);
data_set = replace_func(data_set,'SFoyer',2,16);
data_set = replace_func(data_set,'SLvl',5,16);

data_set = replace_func(data_set,'Flat',4,21);
data_set = replace_func(data_set,'Gable',2,21);
data_set = replace_func(data_set,'Gambrel',1,21);
data_set = replace_func(data_set,'Hip',5,21);
data_set = replace_func(data_set,'Mansard',3,21);
data_set = replace_func(data_set,'Shed',6,21);

% We delete "RoofMatl" because of its low variance.

% data_set = replace_func(data_set,'ClyTile',1,22);
% data_set = replace_func(data_set,'CompShg',2,22);
% data_set = replace_func(data_set,'Membran',3,22);
% data_set = replace_func(data_set,'Metal',4,22);
% data_set = replace_func(data_set,'Roll',5,22);
% data_set = replace_func(data_set,'Tar&Grv',6,22);
% data_set = replace_func(data_set,'WdShake',7,22);
% data_set = replace_func(data_set,'WdShngl',8,22);

data_set = replace_func(data_set,'AsbShng',4,23);
data_set = replace_func(data_set,'AsphShn',2,23);
data_set = replace_func(data_set,'BrkComm',1,23);
data_set = replace_func(data_set,'BrkFace',11,23);
data_set = replace_func(data_set,'CBlock',3,23);
data_set = replace_func(data_set,'CemntBd',13,23);
data_set = replace_func(data_set,'HdBoard',8,23);
data_set = replace_func(data_set,'ImStucc',15,23);
data_set = replace_func(data_set,'MetalSd',6,23);
data_set = replace_func(data_set,'Other',16,23);
data_set = replace_func(data_set,'Plywood',10,23);
data_set = replace_func(data_set,'PreCast',17,23);
data_set = replace_func(data_set,'Stone',14,23);
data_set = replace_func(data_set,'Stucco',9,23);
data_set = replace_func(data_set,'VinylSd',12,23);
data_set = replace_func(data_set,'Wd Sdng',5,23);
data_set = replace_func(data_set,'WdShing',7,23);

data_set = replace_func(data_set,'AsbShng',2,24);
data_set = replace_func(data_set,'AsphShn',4,24);
data_set = replace_func(data_set,'Brk Cmn',3,24);
data_set = replace_func(data_set,'BrkFace',12,24);
data_set = replace_func(data_set,'CBlock',1,24);
data_set = replace_func(data_set,'CmentBd',14,24);
data_set = replace_func(data_set,'HdBoard',11,24);
data_set = replace_func(data_set,'ImStucc',15,24);
data_set = replace_func(data_set,'MetalSd',6,24);
data_set = replace_func(data_set,'Other',16,24);
data_set = replace_func(data_set,'Plywood',10,24);
data_set = replace_func(data_set,'PreCast',17,24);
data_set = replace_func(data_set,'Stone',8,24);
data_set = replace_func(data_set,'Stucco',7,24);
data_set = replace_func(data_set,'VinylSd',13,24);
data_set = replace_func(data_set,'Wd Sdng',5,24);
data_set = replace_func(data_set,'Wd Shng',9,24);

data_set = replace_func(data_set,'BrkCmn',1,25);
data_set = replace_func(data_set,'BrkFace',2,25);
data_set = replace_func(data_set,'CBlock',3,25);
data_set = replace_func(data_set,'None',0,25);
data_set = replace_func(data_set,'Stone',4,25);

data_set = replace_func(data_set,'Ex',5,27);
data_set = replace_func(data_set,'Gd',4,27);
data_set = replace_func(data_set,'TA',3,27);
data_set = replace_func(data_set,'Fa',2,27);
data_set = replace_func(data_set,'Po',1,27);

data_set = replace_func(data_set,'Ex',5,28);
data_set = replace_func(data_set,'Gd',4,28);
data_set = replace_func(data_set,'TA',3,28);
data_set = replace_func(data_set,'Fa',2,28);
data_set = replace_func(data_set,'Po',1,28);

data_set = replace_func(data_set,'BrkTil',2,29);
data_set = replace_func(data_set,'CBlock',3,29);
data_set = replace_func(data_set,'PConc',6,29);
data_set = replace_func(data_set,'Slab',1,29);
data_set = replace_func(data_set,'Stone',4,29);
data_set = replace_func(data_set,'Wood',5,29);

data_set = replace_func(data_set,'Ex',5,30);
data_set = replace_func(data_set,'Gd',4,30);
data_set = replace_func(data_set,'TA',3,30);
data_set = replace_func(data_set,'Fa',2,30);
data_set = replace_func(data_set,'Po',1,30);
data_set = replace_func(data_set,'NA',0,30);

data_set = replace_func(data_set,'Ex',5,31);
data_set = replace_func(data_set,'Gd',4,31);
data_set = replace_func(data_set,'TA',3,31);
data_set = replace_func(data_set,'Fa',2,31);
data_set = replace_func(data_set,'Po',1,31);
data_set = replace_func(data_set,'NA',0,31);

data_set = replace_func(data_set,'Gd',4,32);
data_set = replace_func(data_set,'Av',3,32);
data_set = replace_func(data_set,'Mn',2,32);
data_set = replace_func(data_set,'No',1,32);
data_set = replace_func(data_set,'NA',0,32);

data_set = replace_func(data_set,'GLQ',6,33);
data_set = replace_func(data_set,'ALQ',5,33);
data_set = replace_func(data_set,'BLQ',4,33);
data_set = replace_func(data_set,'Rec',3,33);
data_set = replace_func(data_set,'LwQ',2,33);
data_set = replace_func(data_set,'Unf',1,33);
data_set = replace_func(data_set,'NA',0,33);

data_set = replace_func(data_set,'GLQ',6,35);
data_set = replace_func(data_set,'ALQ',5,35);
data_set = replace_func(data_set,'BLQ',4,35);
data_set = replace_func(data_set,'Rec',3,35);
data_set = replace_func(data_set,'LwQ',2,35);
data_set = replace_func(data_set,'Unf',1,35);
data_set = replace_func(data_set,'NA',0,35);

data_set = replace_func(data_set,'Floor',1,39);
data_set = replace_func(data_set,'GasA',6,39);
data_set = replace_func(data_set,'GasW',5,39);
data_set = replace_func(data_set,'Grav',2,39);
data_set = replace_func(data_set,'OthW',4,39);
data_set = replace_func(data_set,'Wall',3,39);

data_set = replace_func(data_set,'Ex',5,40);
data_set = replace_func(data_set,'Gd',4,40);
data_set = replace_func(data_set,'TA',3,40);
data_set = replace_func(data_set,'Fa',2,40);
data_set = replace_func(data_set,'Po',1,40);

data_set = replace_func(data_set,'N',0,41);
data_set = replace_func(data_set,'Y',1,41);

data_set = replace_func(data_set,'SBrkr',5,42);
data_set = replace_func(data_set,'FuseA',4,42);
data_set = replace_func(data_set,'FuseF',3,42);
data_set = replace_func(data_set,'FuseP',2,42);
data_set = replace_func(data_set,'Mix',1,42);

data_set = replace_func(data_set,'Ex',5,53);
data_set = replace_func(data_set,'Gd',4,53);
data_set = replace_func(data_set,'TA',3,53);
data_set = replace_func(data_set,'Fa',2,53);
data_set = replace_func(data_set,'Po',1,53);

data_set = replace_func(data_set,'Typ',8,55);
data_set = replace_func(data_set,'Min1',7,55);
data_set = replace_func(data_set,'Min2',6,55);
data_set = replace_func(data_set,'Mod',5,55);
data_set = replace_func(data_set,'Maj1',4,55);
data_set = replace_func(data_set,'Maj2',3,55);
data_set = replace_func(data_set,'Sev',2,55);
data_set = replace_func(data_set,'Sal',1,55);

data_set = replace_func(data_set,'Ex',5,57);
data_set = replace_func(data_set,'Gd',4,57);
data_set = replace_func(data_set,'TA',3,57);
data_set = replace_func(data_set,'Fa',2,57);
data_set = replace_func(data_set,'Po',1,57);
data_set = replace_func(data_set,'NA',0,57);

data_set = replace_func(data_set,'2Types',3,58);
data_set = replace_func(data_set,'Attchd',5,58);
data_set = replace_func(data_set,'Basment',4,58);
data_set = replace_func(data_set,'BuiltIn',6,58);
data_set = replace_func(data_set,'CarPort',1,58);
data_set = replace_func(data_set,'Detchd',2,58);
data_set = replace_func(data_set,'NA',0,58);

data_set = replace_func(data_set,'Fin',3,60);
data_set = replace_func(data_set,'RFn',2,60);
data_set = replace_func(data_set,'Unf',1,60);
data_set = replace_func(data_set,'NA',0,60);

data_set = replace_func(data_set,'Ex',5,63);
data_set = replace_func(data_set,'Gd',4,63);
data_set = replace_func(data_set,'TA',3,63);
data_set = replace_func(data_set,'Fa',2,63);
data_set = replace_func(data_set,'Po',1,63);
data_set = replace_func(data_set,'NA',0,63);

data_set = replace_func(data_set,'Ex',5,64);
data_set = replace_func(data_set,'Gd',4,64);
data_set = replace_func(data_set,'TA',3,64);
data_set = replace_func(data_set,'Fa',2,64);
data_set = replace_func(data_set,'Po',1,64);
data_set = replace_func(data_set,'NA',0,64);

data_set = replace_func(data_set,'Y',3,65);
data_set = replace_func(data_set,'P',2,65);
data_set = replace_func(data_set,'N',1,65);

data_set = replace_func(data_set,'WD',5,75);
data_set = replace_func(data_set,'CWD',7,75);
data_set = replace_func(data_set,'VWD',10,75);
data_set = replace_func(data_set,'New',9,75);
data_set = replace_func(data_set,'COD',4,75);
data_set = replace_func(data_set,'Con',8,75);
data_set = replace_func(data_set,'ConLw',3,75);
data_set = replace_func(data_set,'ConLI',6,75);
data_set = replace_func(data_set,'ConLD',2,75);
data_set = replace_func(data_set,'Oth',1,75);

data_set = replace_func(data_set,'Normal',5,76);
data_set = replace_func(data_set,'Abnorml',2,76);
data_set = replace_func(data_set,'AdjLand',1,76);
data_set = replace_func(data_set,'Alloca',4,76);
data_set = replace_func(data_set,'Family',3,76);
data_set = replace_func(data_set,'Partial',6,76);

% It was decided to delete the following controversial features and data samples
% The follwoing features had low variance
data_set(:,71) = []; % "PoolArea"
data_set(:,69) = []; % "3SsnPorch"
data_set(:,45) = []; % "LowQualFinSF"
data_set(:,39) = []; % "Heating"
data_set(:,22) = []; % "RoofMatl"
data_set(:,14) = []; % "Condition2"
data_set(:,9) = []; % "Utilities"
data_set(:,6) = []; % "Street"

data_set(1299,:) = [];
data_set(1191,:) = [];
data_set(1183,:) = [];
data_set(1062,:) = [];
data_set(935,:) = [];
data_set(707,:) = [];
data_set(692,:) = [];
data_set(582,:) = [];
data_set(379,:) = [];
data_set(336,:) = [];
data_set(314,:) = [];
data_set(250,:) = [];

% All the missing values that were not deleted are converted into NaN
% values so it would be easier to work with the array
for i = 1:size(data_set,1)
    for j = 1:size(data_set,2)
        a = data_set{i,j}; % Checking every value of the array
        if length(a) == length('NA') % Checking the length of the value to avoid any mistakes
            if a(1:length('NA')) == 'NA' % Checking if first two values correspond to N and A
                data_set{i,j} = NaN; % Change it to NaN
            end
        end
    end
end

% Create an empty array of the same size as data_set
data = zeros(size(data_set,1),size(data_set,2));

% One by one each feature (column) converted from "char" to "double"
for i = 1:size(data_set,2)
    temp = class(data_set{1,i}); % Temporary value that stors the type of data in this column
    if temp(1:4) == 'char' % Check if the type is "char"
        data(:,i) = str2double(data_set(:,i)); % The whole column converted into numeric values and stored in "data"
        continue; % Skipping next two lines and going for the next iteration
    end
    % Some "char" values were already encoded as numeric values so, if 
    % the data type is not "char", MATLAB can convert the whole column of
    % cells into numeric vector
    data(:,i) = cell2mat(data_set(:,i)); % The values are converted and stored in "data"
end

% Some features like "LotFrontage" or "MasVnrArea" contain missing values
% It was decided to find these values and change them to 0
NaN_change = [false(size(data,1),3) isnan(data(:,4)) false(size(data,1),16) isnan(data(:,21)) isnan(data(:,22))];
data(NaN_change) = 0;

% For other features like "MSZoning" or "Functional" it was decided to
% change the missing values to the most frequent values of this feature
[row, col] = find(isnan(data)); % Find indexes of missing values
A = mode(data); % Find the most frequent values of each column

for i = 1:180
    data(row(i),col(i)) = A(col(i)); % Change all missing values to the most frequent values of the corresponding features
end

% Now data is clean. There are no missing values or low variance features

% Training data set is split into actual training data and testing data
data_tr_tr = data(1:1250,2:68); % 86% of data will be our training data
label_tr_tr = data(1:1250,69); % Labels of training data

data_tr_te = data(1251:1448,2:68); % The rest 14% is our testing data
label_tr_te = data(1251:1448,69); % The labels of testing data

data_output = data(1449:end,2:68); % This data set contains the samples which "SalePrice" must be predicted for the Kaggle challenge

% Some features contain values that vary in a wide range
% This kind of data should be normalized
for i = [4 29 31 38 59 60 61 62 63] % Selecting these features
    xi = data_tr_tr(:,i); % Saving only selected feature
    mi = mean(xi); % Calculating mean for this column
    vi = sqrt(var(xi)); % Calculating variance
    data_tr_tr(:,i) = (xi - mi)/vi; % Applying it to each data sample of the selected feature
end

% The same process is applied to testing dataset and challenge dataset
for i = [4 29 31 38 59 60 61 62 63]
    xi = data_tr_te(:,i);
    mi = mean(xi);
    vi = sqrt(var(xi));
    data_tr_te(:,i) = (xi - mi)/vi;
end

for i = [4 29 31 38 59 60 61 62 63]
    xi = data_output(:,i);
    mi = mean(xi);
    vi = sqrt(var(xi));
    data_output(:,i) = (xi - mi)/vi;
end

% Next steps describe linear regression approach
data_tr_tr_hat = [data_tr_tr ones(1250,1)];

% Compute coefficients w and b that are included in w_hat
% It is important to include 0.01*diag(ones(68)) to assure that the inverse
% of the matrix does exist
w_hat = (data_tr_tr_hat'*data_tr_tr_hat+0.01*diag(ones(68)))\(data_tr_tr_hat'*label_tr_tr);

w = w_hat(1:67); % Extracting w
b = w_hat(68); % Extracting b

Y = w'*data_tr_te' + b; % Calculate the output values using linear model
Y = [data(1251:1448,1) Y']; % Include "ID"s for the output

challenge_output = w'*data_output' + b; % Calculate the output values for the challenge
challenge_output = [data(1449:end,1) challenge_output']; % Include "ID"s for the output

RMSLE = sqrt(mean((log(label_tr_te)-log(Y(:,2))).^2)); % Calculate RMSLE
error = (norm(label_tr_te-Y(:,2),'fro')/norm(label_tr_te,'fro')); % Calculate overall relative prediction error

plot(Y(:,1),Y(:,2),'b','linew',1.5) % Plot predicted output
hold on
plot(Y(:,1),label_tr_te,'r','linew',1.5) % Plot actual labels
xlabel('ID')
ylabel('SalePrice')
xlim([1260 1460])
ylim([50000 500000])
grid on
grid minor
legend('Predicted values', 'Actual values')

challenge_output = ["Id" "SalePrice"; challenge_output];
writematrix(challenge_output,'submission.csv')
