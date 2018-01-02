clear;
%Set Up Data%
m1 = [0; 3];
C1 = [2 1; 1 2];

m2 = [2; 1];
C2 = C1;

N = 1000;

X1 = mvnrnd(m1, C1, N);
X2 = mvnrnd(m2, C2, N);

X = [X1; X2];

dimOfData = size(X);
dimOfData = dimOfData(1, 2);

for i=1: dimOfData
    X(:, i) = X(:,i) - mean(X(:,i));
    X(:, i) = X(:, i) / std(X(:, i));
end
X1 = X(1:N, :);
X2 = X(N+1: N*2, :);

plot(X1(:,1),X1(:,2),'c.', X2(:,1),X2(:,2),'mx');
hold on;

Y = [zeros(N,1); ones(N,1)]; 

%Bayes Solution
trueWeight = inv(C1) * (m2 - m1) * 2;

low = min(X);
lowX = low(1,1);
lowY = low(1,2);

high = max(X);
highX = high(1,1);
highY = high(1,2);
axis([lowX highX lowY highY]); 

a1 = ((lowX * trueWeight(1,1)) / trueWeight(2,1)) * - 1;
a2 = ((highX * trueWeight(1,1)) / trueWeight(2,1)) * - 1;

ii = randperm(N*2);

Xtr = X(ii(1:N), :);
Xtst = X(ii(N+1: N*2), :);

Ytr = Y(ii(1:N), :);
Ytst = Y(ii(N+1: N*2), :);

%Generic Particle Filter%
noWeights = 1000;
noWeights100 = 100;

weights = zeros(dimOfData, noWeights);
impWeights = zeros(noWeights, 1);

weights100 = zeros(dimOfData, noWeights100);
impWeights100 = zeros(noWeights100, 1);

for i=1:noWeights
    weights(:, i) = mvnrnd(-1, 10, dimOfData);
    impWeights(i, 1) = (1 / noWeights);
end

for i=1:noWeights100
    weights100(:, i) = mvnrnd(0, 10, dimOfData);
    impWeights100(i, 1) = (1 / noWeights100);
end

%To show the motion of particles
weights1Ratio = weights(1, :) ./ weights(2, :);
weights100Ratio = zeros(noWeights, 1);
weights250Ratio = zeros(noWeights, 1);
weights500Ratio = zeros(noWeights, 1);
weights750Ratio = zeros(noWeights, 1);
weights1000Ratio = zeros(noWeights, 1);

%Initialise variabels
noStates = 1000; %State count
diffWeights = zeros(noStates, 1); %1000 Weights, holds the ratio of the weights at each state
diffWeights100 = zeros(noStates, 1); %100 Weights

Q = 1; %StateNoise%
R = 10; %MeasurementNoise%

for k=1:noStates
    %Training Process
    r = ceil(rand*N);
    currentX = Xtr(r, :);
    currentY = Ytr(r, 1);
    for i=1:noWeights
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
    for i = 1:noWeights
        weightsSum = weightsSum + impWeights(i, 1);
        weightsSumSquared = weightsSumSquared + impWeights(i, 1)^2;
    end
    impWeights = impWeights / weightsSum;
    

    eff = 1 / weightsSumSquared;
    if eff < noWeights/3
        weightsTemp = weights;
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
            weights(:, j) = weightsTemp(:, count);
        end
        impWeights = impWeightsTemp / noWeights;
    end
    
    %Same process as above but for 100 weights
     for i=1:noWeights100
        dk = mvnrnd(0, Q, dimOfData);
        currentWeight = weights100(:, i);
        currentImpWeight = impWeights100(i, 1);
        currentWeight = currentWeight + dk;
        weights100(:, i) = currentWeight;
        
        logisticOutput = 1 / (1 + exp(-(currentX*currentWeight)));
        likelihood = (exp(-1 * (((currentY - logisticOutput)^2) / 2*R))) / ((2*R*pi)^0.5);
        impWeight = currentImpWeight * likelihood;
        impWeights100(i, 1) = impWeight;
    end
    
    weightsSum = 0;
    weightsSumSquared = 0;
    for i = 1:noWeights100
        weightsSum = weightsSum + impWeights100(i, 1);
        weightsSumSquared = weightsSumSquared + impWeights100(i, 1)^2;
    end
    impWeights100 = impWeights100 / weightsSum;
    

    eff = 1 / weightsSumSquared;
    if eff < noWeights100/3
        weightsTemp = weights100;
        impWeightsTemp = ones(noWeights100, 1);
        cdf = zeros(noWeights100+1, 1);
        cdf(1,1) = 0;
        for i = 2:noWeights100+1
            cdf(i, 1) = cdf(i-1, 1) + impWeights100(i-1, 1);
        end
        for j = 1:noWeights100
            count = 1;
            i = rand;
            while i > cdf(count + 1, 1)
                count = count + 1;
            end
            weights100(:, j) = weightsTemp(:, count);
        end
        impWeights100 = impWeightsTemp / noWeights100;
    end
   

    if k == 100
        weights100Ratio = weights(1, :) ./ weights(2, :);
    elseif k == 250
        weights250Ratio = weights(1, :) ./ weights(2, :);
    elseif k == 500
        weights500Ratio = weights(1, :) ./ weights(2, :);
    elseif k == 750
        weights750Ratio = weights(1, :) ./ weights(2, :);
    elseif k == 1000
        weights1000Ratio = weights(1, :) ./ weights(2, :);
    end
        
    
    finalWeight = weights * impWeights;
    ratio = finalWeight(1,1) / finalWeight(2,1);
    diffWeights(k, 1) = ratio;
    
    finalWeight100 = weights100 * impWeights100;
    ratio100 = finalWeight100(1,1) / finalWeight100(2,1);
    diffWeights100(k, 1) = ratio100;
    
    y1 = ((lowX * finalWeight(1,1)) / finalWeight(2,1)) * - 1;
    y2 = ((highX * finalWeight(1,1)) / finalWeight(2,1)) * - 1;
    draw(lowX, highX, y1, y2); %For animation
end

drawBoundary(lowX, highX, y1, y2, a1, a2);
drawConvergence(diffWeights, diffWeights100);
drawRatioMotion(noStates, trueWeight, weights1Ratio, weights100Ratio, weights250Ratio, weights500Ratio, weights750Ratio, weights1000Ratio);
displayAccuracy(Xtst, Ytst, finalWeight, trueWeight, N);

function draw(lowX, highX, y1, y2)
    l = plot([lowX highX], [y1 y2], 'b', 'LineWidth', 2);
    drawnow;
    pause(0.01);
    delete(l);
end

function drawBoundary(lowX, highX, y1, y2, a1, a2)
     l = plot([lowX highX], [y1 y2], 'b', 'LineWidth', 2); %Trained Solution
     hold on;
     l2 = plot([lowX highX], [a1 a2], 'k', 'LineWidth', 2); %True Solution
     figure;
end

function drawConvergence(diffWeights, diffWeights100)
    plot(log(diffWeights.^2), 'Color', 'k');
    hold on;
    plot(log(diffWeights100.^2), 'Color', 'b');
    figure;
end

function drawRatioMotion(noStates, trueWeight, weights1Ratio, weights100Ratio, weights250Ratio, weights500Ratio, weights750Ratio, weights1000Ratio)
    x = 0 : noStates;
    trueRatio = trueWeight(1,1) / trueWeight(2,1);
    trueRatio = log(trueRatio^2);
    plot(x, trueRatio, 'k.');
    hold on;
    weights1Ratio = log(weights1Ratio.^2);
    plot(1, weights1Ratio, 'k.');
    hold on;
    weights100Ratio = log(weights100Ratio.^2);
    plot(100, weights100Ratio, 'k.');
    hold on;
    weights250Ratio = log(weights250Ratio.^2);
    plot(250, weights250Ratio, 'k.');
    hold on;
    weights500Ratio = log(weights500Ratio.^2);
    plot(500, weights500Ratio, 'k.');
    hold on;
    weights750Ratio = log(weights750Ratio.^2);
    plot(750, weights750Ratio, 'k.');
    hold on;
    weights1000Ratio = log(weights1000Ratio.^2);
    plot(1000, weights1000Ratio, 'k.');
end

function displayAccuracy(Xtst, Ytst, finalWeight, trueWeight, N)
    correct = 0;
    correctTrue = 0;

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

        predY = currentX * trueWeight;
        if predY >= 0
            predY = 1;
        else
            predY = 0;
        end
        if predY == currentY
            correctTrue = correctTrue + 1;
        end
    end

    accuracy = correct / N;
    accuracyTrue = correctTrue / N;
end
    
