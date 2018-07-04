%--- A a
function dimensionVectors = threeDSpace(model, trainVect)
  dimensionVectors = [];

  sizeW = size(model.weight,1);
  sizeV = size(trainVect,1);

  for i = 1:sizeW
      clusters =[];
      cov = squeeze(model.covar(i,:,:));
      for j = 1:sizeV
          logProbality = log(model.weight(i))-0.5*(log(det(cov))+(ctranspose(ctranspose(trainVect(j,:))-ctranspose(model.mean(i,:))))*inv(cov)*(ctranspose(trainVect(j,:))-(ctranspose(model.mean(i,:)))));
          clusters = [clusters,logProbality];
      end
      dimensionVectors = [LnVectorProb;clusters];
  end
end
