%--- A c
function model = implementKmean(model, trainVect, componentProbalityLine)
  for i = 1:size(componentProbalityLine,1)
    exSum = sum(exp(componentProbalityLine(i,:)));  %sum of the exponential values in task b
    featureVectNum = size(componentProbalityLine,2);

    % weight of cluster i
    model.weight(i) = exSum/featureVectNum;
    actualCluster = 0;
    j = 1;
    %loop of feature vectors
    while(j <= featureVectNum)
      centroid = trainVect(j,:) * exp(featureVectNum(i,j));
      actualCluster = actualCluster + centroid;
      j = j + 1;
    end
    actualCluster = actualCluster*(1/exSum);
    model.mean(i,:) = actualCluster;
    covMatrix = 0;
    j = 1;
    while(j <= featureVectNum)
      cov = (ctranspose(trainVect(j,:))-ctranspose(actualCluster))*ctranspose(ctranspose(trainVect(j,:))-ctranspose(actualCluster))*exp(componentProbalityLine(i,j));
      covMatrix = covMatrix + cov;
      j = j + 1;
    end
    covMatrix = covMatrix*(1/exSum);
    model.covar(i,:,:) = covMatrix;
  end
end
