function model = gaussianMixtureModel(trainVect,componentNumb)
  model.weight(1,:)=1;
  model.mean=[0,0,0];
  model.covar(1,:,:)=[1 0 0; 0 1 0; 0 0 1];
  stp = 10^-6;
  for i=1:
    lastPx=-inf;
    TotalProbabLine = caculatetotalProbability(model, trainVect);

    while(TotalProbabLine-lastPx>stp)
      componentProbalityLine = kMeanClustering(model, trainVect);
      model = mplementKmean(model, trainVect, componentProbalityLine);
      TotalProbabLine = caculatetotalProbability(model, trainVect);
    end
    clf
    PlotGMM(model,trainVect);
    drawnow;

    if i<componentNumb
      model = InitNewComponent(model, trainVect);
    end
  end
end

function totalProbabLine = caculatetotalProbability(model, trainVect)
  componentNumb = numel(model.weight);
  dimensionVectors = threeDSpace(model, trainVect);
  maxDimensionVector = max(dimensionVectors,[],1);
  scalingFactors = repmat(maxDimensionVector,componentNumb,1);  %resize to totalProbabLine

  % scaling of logarithmic probabilities before using exp in order
  newDimensionVectors = maxDimensionVector + log(sum(exp(dimensionVectors - scalingFactors),1));
  totalProbabLine = sum(newDimensionVectors);
end

function PlotGMM(model, trainVect)
  componentNumb = numel(model.weight);
  dimensionNumb = size(trainVect,2);

  hold on;

  %plot feature points
  plot3(trainVect(:,1),trainVect(:,2),trainVect(:,3), 'g.','MarkerSize',7);

  % plot elements of the estimated components:
  for i=1:componentNumb

      % eigenvektor / eigenwert decomposition
      [eVec,eVal] = eig(squeeze(model.covar(i,:,:)));

      % plotting of mean values
      mean = squeeze(model.mean(i,:));
      plot3(mean(1),mean(2),mean(3),'ro');

      % derivation and plotting
      for i=1:dimensionNumb
          devVec = (sqrt(eVal(i,i)) * eVec(:,i))*[-1,1];
          plot3(mean(1) + devVec(1,:), mean(2) + devVec(2,:), mean(3) + devVec(3,:),'b');
      end
  end

  % rotate 3D view and setting
  hold off;
  view([19,25]);
  grid('on');
  title(['Gaussian Mixture Model (',num2str(componentNumb),' components)'])

end

function NewModel = InitNewComponent(model, trainVect)
  componentNumb = numel(model.weight);
  dimensionNumb = size(trainVect, 2);

  [ignore, splitComp] = max(model.weight);

  % calculate new weight vector, mean and covariance
  newWeight = zeros(componentNumb+1,1);
  newMean = zeros(componentNumb+1,dimensionNumb);
  newCovar = zeros(componentNumb+1,dimensionNumb,dimensionNumb);

  % copy old values into new arrays
  newWeight(1:componentNumb) = model.weight;
  newMean(1:componentNumb,:) = model.mean;
  newCovar(1:componentNumb,:,:) = model.covar;

  % Component splitComp will be splitted along its dominant axis
  [eVec,eVal] = eig(squeeze(newCovar(splitComp,:,:)));
  [ignore, majAxis] = max(diag(eVal));
  devVec = sqrt(eVal(majAxis,majAxis)) * eVec(:,majAxis)';

  % initialize new component
  % half of the points: half weight
  newWeight(componentNumb+1) = 0.5*newWeight(splitComp);
  % shift new mean to half of length along dominant axis
  newMean(componentNumb+1,:) = newMean(splitComp,:) - 0.5*devVec;
  % make covariance a little bit smaller
  newCovar(componentNumb+1,:,:) = newCovar(splitComp,:,:) / 4.0;

  % update also the (old) splitted component
  % also half of the points
  newWeight(splitComp) = newWeight(componentNumb+1);
  % shift comonent center to other direction along dominant axis
  newMean(splitComp,:) = newMean(componentNumb+1,:) + devVec;
  % take same smaller covariance matrix
  newCovar(splitComp,:,:) = newCovar(componentNumb+1,:,:);

  % store new parameters in model
  NewModel.weight = newWeight;
  NewModel.mean = newMean;
  NewModel.covar = newCovar;
end
