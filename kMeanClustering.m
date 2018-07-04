%--- A b
function componentProbalityLine = kMeanClustering(model, trainVect)
  dimensionVectors = threeDSpace(model, trainVect);
  componentProbalityLine = zeros(size(dimensionVectors));
  denominator = []; %holding enominator
  clusterNum = size(dimensionVectors,1);
  featureVectNum = size(dimensionVectors,2);

  for a = 1:featureVectNum
    denominator = [denominator,log(sum(exp(dimensionVectors(:,a))));];
  end
  
  for a = 1:clusterNum
    for b = 1:featureVectNum
      componentProbalityLine(a,b) = dimensionVectors(a,b) - denominator(1,b);
    end
  end

end
