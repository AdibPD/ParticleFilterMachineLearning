clear;
TitanicTrainingData = csvread('C:\Users\AdibPD\Downloads\Titanic Data\TitanicTrainingData.csv');

output = TitanicTrainingData(:, 2);
input = TitanicTrainingData(:, 3:5);
N = length(output);

dimOfData = size(input);
dimOfData = dimOfData(1, 2);

for i=1: dimOfData
    input(:, i) = input(:,i) - mean(input(:,i));
    input(:, i) = input(:, i) / std(input(:, i));
end

%Particle Filter%
noWeights = 1000;

weights = zeros(dimOfData, noWeights);
impWeights = zeros(noWeights, 1);

for i=1:noWeights
    weights(:, i) = randn(dimOfData, 1);
    impWeights(i, 1) = (1 / noWeights);
end

noStates = 1000;

Q = 1; %StateNoise%
R = 10; %MeasurementNoise%

for k=1:noStates
    r = ceil(length(input));
    currentX = input(r, :);
    currentY = output(r, 1);
    
    weightsTemp = weights;
    for i=1:noWeights
        dk = mvnrnd(0, Q, dimOfData);
        currentWeight = weights(:, i);
        currentImpWeight = impWeights(i, 1);
        currentWeight = currentWeight + dk;
        weightsTemp(:, i) = currentWeight;
        
        logisticOutput = 1 / (1 + exp(-(currentX*currentWeight)));
        likelihood = (exp(-1 * (((currentY - logisticOutput)^2) / 2*R))) / ((2*R*pi)^0.5);
        impWeight = currentImpWeight * likelihood;
        impWeights(i, 1) = impWeight;
    end
    
    weightsSum = 0;
    for i = 1:noWeights
        weightsSum = weightsSum + impWeights(i, 1);
    end
    impWeights = impWeights / weightsSum;
    
    index = impWeights;
    weightsTemp2 = weightsTemp;
    impWeightsTemp = ones(noWeights, 1);
    cdf = zeros(noWeights+1, 1);
    cdf(1,1) = 0;
    for i = 2:noWeights+1
        cdf(i, 1) = cdf(i-1, 1) + impWeights(i-1, 1);
    end
    for j = 1:noWeights
        count = 1;
        i = rand;
        while i > cdf(count + 1, 1)
            count = count + 1;
        end
        weightsTemp(:, j) = weightsTemp2(:, count);
        index(j, 1) = count;
    end
    impWeights = impWeightsTemp / noWeights;
    
    for i=1:noWeights
        dk = mvnrnd(0, Q, dimOfData);
        currentWeight = weights(:, index(i, 1));
        currentImpWeight = impWeights(i, 1);
        currentWeight = currentWeight + dk;
        weights(:, i) = currentWeight;
        
        logisticOutput = 1 / (1 + exp(-(currentX*currentWeight)));
        likelihood = (exp(-1 * (((currentY - logisticOutput)^2) / 2*R))) / ((2*R*pi)^0.5);
        impWeight = likelihood / impWeights(index(i,1), 1);
        impWeights(i, 1) = impWeight;
    end
    
    weightsSum = 0;
    for i = 1:noWeights
        weightsSum = weightsSum + impWeights(i, 1);
    end
    impWeights = impWeights / weightsSum;
    
    finalWeight = weights * impWeights;
end

TitanicTestingData = csvread('C:\Users\AdibPD\Downloads\Titanic Data\TitanicTestingData.csv');
inputTest = TitanicTestingData(:, 2:4);
for i=1: dimOfData
    inputTest(:, i) = inputTest(:,i) - mean(inputTest(:,i));
    inputTest(:, i) = inputTest(:, i) / std(inputTest(:, i));
end
outputTest = inputTest * finalWeight;
predictionsTest = 1 ./ (1 + exp(-outputTest));
predictionsHalfThresholdTest = predictionsTest;
predictionsHalfThresholdTest(predictionsTest >= 0.5) = 1;
predictionsHalfThresholdTest(predictionsTest < 0.5) = 0;
finalOutput = [TitanicTestingData(:,1), predictionsHalfThresholdTest];

fid = fopen('C:\Users\AdibPD\Downloads\Titanic Data\gender_submission.csv', 'w');
fprintf(fid, 'PassengerId,Survived\n');
fclose(fid);

dlmwrite('C:\Users\AdibPD\Downloads\Titanic Data\gender_submission.csv', finalOutput, '-append');