clear;

data = fileread('Data.txt');
data = strsplit(data,'\n');
data = data';

sizeOfData = length(data) - 1;
X1 = zeros(1, 2);
X2 = zeros(1,2);
x1counter = 1;
x2counter = 1;
output = zeros(sizeOfData, 1);
for i=1:sizeOfData
    currentData = cell2mat(data(i));
    currentData = strsplit(currentData, ' ');
    x1 = str2double(currentData(1));
    x2 = str2double(currentData(2));
    y = str2double(currentData(3));
    if(y == 0)
      X1(x1counter, 1) = x1;  
      X1(x1counter, 2) = x2;
      x1counter = x1counter + 1;
    else 
       X2(x2counter, 1) = x1;
       X2(x2counter, 2) = x2;
       x2counter = x2counter + 1;
    end
end


X = [X1; X2];
Y = [zeros(length(X1),1); ones(length(X2),1)]; 

dimOfData = size(X);
dimOfData = dimOfData(1, 2);

for i=1: 2
    X(:, i) = X(:,i) - mean(X(:,i));
    X(:, i) = X(:, i) / std(X(:, i));
end

plot(X(1:length(X1),1),X(1:length(X1),2),'c.', X(length(X1)+1:length(X),1),X(length(X1)+1:length(X),2),'mx');
hold on;

low = min(X);
lowX = low(1,1);
lowY = low(1,2);

high = max(X);
highX = high(1,1);
highY = high(1,2);
axis([lowX highX lowY highY]); 

ii = randperm(sizeOfData);

Xtr = X(ii(1:floor(sizeOfData/2)), :);
Xtst = X(ii((floor(sizeOfData/2)) + 1: sizeOfData), :);

Ytr = Y(ii(1:floor(sizeOfData/2)), :);
Ytst = Y(ii(floor(sizeOfData/2) + 1: sizeOfData), :);

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
    y1 = ((lowX * finalWeight(1,1)) / finalWeight(2,1)) * - 1;
    y2 = ((highX * finalWeight(1,1)) / finalWeight(2,1)) * - 1;
end
    
l = plot([lowX highX], [y1 y2], 'b', 'LineWidth', 2);


predictions = Xtst * finalWeight;
predictions = 1 ./ (1 + exp(-predictions));
predictionsHalfThreshold = predictions;
predictionsHalfThreshold(predictions >= 0.5) = 1;
predictionsHalfThreshold(predictions < 0.5) = 0;
correct = length(find(predictionsHalfThreshold == Ytst));
accuracy = correct / length(Xtst);

drawROCCurve(predictions, Ytst)

function a = drawROCCurve(predictions, Ytst)
    thmin = 0;
    thmax = 1;
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
    a = area = trapz(ROC(:,1), ROC(:,2)) * -1;
end

    

    



