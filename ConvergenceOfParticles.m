clear;
%Set Up Data%
m1 = [0; 3];
C1 = [2 1; 1 2];

m2 = [2; 1];
C2 = [1 0; 0 1];

N = 1000;

X1 = mvnrnd(m1, C1, N);
X2 = mvnrnd(m2, C2, N);

plot(X1(:,1),X1(:,2),'c.', X2(:,1),X2(:,2),'mx');
hold on;

X = [X1; X2];
X = [X ones(N*2, 1)];
Y = [zeros(N,1); ones(N,1)]; 

dimOfData = size(X);
dimOfData = dimOfData(1, 2);

low = min(X);
lowX = low(1,1);
lowY = low(1,2);

high = max(X);
highX = high(1,1);
highY = high(1,2);
axis([lowX highX lowY highY]); 

ii = randperm(N*2);

Xtr = X(ii(1:N), :);
Xtst = X(ii(N+1: N*2), :);

Ytr = Y(ii(1:N), :);
Ytst = Y(ii(N+1: N*2), :);

%Particle Filter%
noWeightsOne = 100;
noWeightsTwo = 1000;

weights = zeros(dimOfData, noWeightsOne);
impWeights = zeros(noWeightsOne, 1);

weightsTwo = zeros(dimOfData, noWeightsTwo);
impWeightsTwo = zeros(noWeightsTwo, 1);

for i=1:noWeightsOne
    weights(:, i) = randn(dimOfData, 1);
    impWeights(i, 1) = (1 / noWeightsOne);
end

for i=1:noWeightsTwo
    weightsTwo(:, i) = randn(dimOfData, 1);
    impWeightsTwo(i, 1) = (1 / noWeightsOne);
end

noStates = 1000;
diffWeights = zeros(noStates, 1);
diffWeightsTwo = zeros(noStates, 1);

weightOne = zeros(noStates, 1);
weightTwo = zeros(noStates, 1);
weightThree = zeros(noStates, 1);

Q = 1; %StateNoise%
R = 10; %MeasurementNoise%

for k=1:noStates
    r = ceil(rand*N);
    currentX = Xtr(r, :);
    currentY = Ytr(r, 1);
    
    for i=1:noWeightsOne
        dk = mvnrnd(0, Q, dimOfData);
        currentWeight = weights(:, i);
        currentImpWeight = impWeights(i, 1);
        currentWeight = currentWeight + dk;
        weights(:, i) = currentWeight;
        
        logisticOutput = 1 / (1 + exp(-(currentX*currentWeight)));
        likelihood = (exp(-1 * (((currentY - logisticOutput)^2) / 2*R))) / ((2*R*pi)^0.5);
        impWeight = currentImpWeight * likelihood;
        impWeights(i, 1) = impWeight;
    end
    
    weightsSum = 0;
    weightsSumSquared = 0;
    for i = 1:noWeightsOne
        weightsSum = weightsSum + impWeights(i, 1);
        weightsSumSquared = weightsSumSquared + impWeights(i, 1)^2;
    end
    impWeights = impWeights / weightsSum;
    
    for i=1:noWeightsTwo
        dk = mvnrnd(0, Q, dimOfData);
        currentWeight = weightsTwo(:, i);
        currentImpWeight = impWeightsTwo(i, 1);
        currentWeight = currentWeight + dk;
        weightsTwo(:, i) = currentWeight;
        
        logisticOutput = 1 / (1 + exp(-(currentX*currentWeight)));
        likelihood = (exp(-1 * (((currentY - logisticOutput)^2) / 2*R))) / ((2*R*pi)^0.5);
        impWeight = currentImpWeight * likelihood;
        impWeightsTwo(i, 1) = impWeight;
    end
    
    weightsSum = 0;
    weightsSumSquared = 0;
    for i = 1:noWeightsTwo
        weightsSum = weightsSum + impWeightsTwo(i, 1);
        weightsSumSquared = weightsSumSquared + impWeightsTwo(i, 1)^2;
    end
    impWeightsTwo = impWeightsTwo / weightsSum;
    

%     eff = 1 / weightsSumSquared;
%     if eff < noWeights/3
%         weightsTemp = weights;
%         impWeightsTemp = ones(noWeights, 1);
%         cdf = zeros(noWeights+1, 1);
%         cdf(1,1) = 0;
%         for i = 2:noWeights+1
%             cdf(i, 1) = cdf(i-1, 1) + impWeights(i-1, 1);
%         end
%         for j = 1:noWeights
%             count = 1;
%             i = rand;
%             if i < cdf(count + 1, 1)
%                 weights(:, j) = weightsTemp(:, count);
%             end
%         end
%         impWeights = impWeightsTemp / noWeights;
%     end
    
    
    finalWeight = weights * impWeights;
    diffWeights(k, 1) = norm(finalWeight);
    
    finalWeightTwo = weightsTwo * impWeightsTwo;
    diffWeightsTwo(k, 1) = norm(finalWeightTwo);
    
    weightOne(k, 1) = finalWeightTwo(1,1);
    weightTwo(k, 1) = finalWeightTwo(2, 1);
    weightThree(k, 1) = finalWeightTwo(3, 1);
    
    y1 = (((lowX * finalWeightTwo(1,1)) + finalWeightTwo(3, 1)) / finalWeightTwo(2,1)) * - 1;
    y2 = (((highX * finalWeightTwo(1,1)) + finalWeightTwo(3, 1)) / finalWeightTwo(2,1)) * - 1;
%     l = plot([lowX highX], [y1 y2], 'b', 'LineWidth', 2);
%     hold on;
    a1 = (((lowX * finalWeight(1,1)) + finalWeight(3, 1)) / finalWeight(2,1)) * - 1;
    a2 = (((highX * finalWeight(1,1)) + finalWeight(3, 1)) / finalWeight(2,1)) * - 1;
%     l2 = plot([lowX highX], [a1 a2], 'k', 'LineWidth', 2);
%     
%     drawnow;
%     pause(0.01);
%     delete(l);
%     delete(l2);
    
end
l = plot([lowX highX], [y1 y2], 'b', 'LineWidth', 2);
hold on;
l2 = plot([lowX highX], [a1 a2], 'k', 'LineWidth', 2);
figure;
plot(diffWeights);
hold on;
plot(diffWeightsTwo);
figure;

plot(weightOne);
hold on;
plot(weightTwo);
hold on;
plot(weightThree);

correct = 0;
correct2 = 0;

for i = 1: N
    currentX = Xtst(i, :);
    currentY = Ytst(i, 1);
    predY = currentX * finalWeight;
    if predY >= 0
        predY = 1;
    else
        predY = 0;
    end
    if predY == currentY
        correct = correct + 1;
    end
    
    predY2 = currentX * finalWeightTwo;
    if predY2 >= 0
        predY2 = 1;
    else
        predY2 = 0;
    end
    if predY2 == currentY
        correct2 = correct2 + 1;
    end
end

accuracy = correct / N;
accuracy2 = correct2 / N;
    

