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

ii = randperm(N);

Xtr = input(ii(1:floor(N/2)), :);
Xtst = input(ii(floor(N/2) + 1: N), :);

Ytr = output(ii(1:floor(N/2)), :);
Ytst = output(ii(floor(N/2) + 1: N), :);


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
    r = ceil(length(Xtr));
    currentX = Xtr(r, :);
    currentY = Ytr(r, 1);
    
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

predictions = Xtst * finalWeight;
predictions = 1 ./ (1 + exp(-predictions));
predictionsHalfThresholdTest = predictions;
predictionsHalfThresholdTest(predictions >= 0.5) = 1;
predictionsHalfThresholdTest(predictions < 0.5) = 0;
correct = length(find(predictionsHalfThresholdTest == Ytst));
accuracy = correct / length(Xtst);

drawROCCurve(predictions, Ytst)
writeResultsToFile(finalWeight, dimOfData);

function a = drawROCCurve(predictions, Ytst)
    thmin = 1.1;
    thmax = 0;
    rocResolution = 100;
    thRange = linspace(thmin, thmax, rocResolution);
    ROC = zeros(rocResolution,2);

    for jThreshold = 1: rocResolution
        threshold = thRange(jThreshold);
        predictionsTemp = predictions;
        predictionsTemp(predictions >= threshold) = 1;
        predictionsTemp(predictions < threshold) = 0;
        matches = find(predictionsTemp == Ytst);
        matches = predictionsTemp(matches);
        truePositive = length(find(matches == 1));
        trueNegative = length(find(matches == 0));

        falsePositive = length(find(predictionsTemp == 1)) - truePositive;
        falseNegative = length(find(predictionsTemp == 0)) - trueNegative;

        sensitivity = truePositive / (truePositive + falseNegative);
        specificity = trueNegative / (trueNegative + falsePositive);

        ROC(jThreshold,:) = [(1-specificity) sensitivity];
    end


    figure(3), clf,
    plot(ROC(:,1), ROC(:,2), 'b', 'LineWidth', 2);
    axis([0 1 0 1]);
    grid on, hold on
    plot(0:1, 0:1, 'k-');
    xlabel('False Positive', 'FontSize', 16)
    ylabel('True Positive', 'FontSize', 16);
    title('Receiver Operating Characteristic Curve', 'FontSize', 20);
    area = trapz(ROC(:,1), ROC(:,2));
    a = area;
end

function writeResultsToFile(finalWeight, dimOfData)
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
    fprintf(fid, 'PassengerId, Survived\n');
    fclose(fid);

    dlmwrite('C:\Users\AdibPD\Downloads\Titanic Data\gender_submission.csv', finalOutput, '-append');
end
