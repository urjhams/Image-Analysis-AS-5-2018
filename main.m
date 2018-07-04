function main()
  img = imread("inputEx5.jpg");
  image = mat2gray(img);
  imgSize = size(image);

  dimensionNumb = imgSize(3);
  componentNumb = 10;

  %reshape
  trainVect = reshape(image, [imgSize(1) * imgSize(2),dimensionNumb]);

  desireNumb = 1000;
  step = imgSize(1)*imgSize(2)/desireNumb;
  indices = round(1:step:imgSize(1)*imgSize(2));
  trainVect = trainVect(indices,:);
  edgePoints = sum(trainVect,2);
  test = find(edgePoints ~= 3.0);
  trainVect = trainVect(test,:);

  model = gaussianMixtureModel(trainVect,componentNumb);

  classedImg = classifyImage(model, image);

  figure(2);
  subplot(1,2,1), imshow(image), title('Original Image');
  subplot(1,2,2), imshow(classedImg,[]), colormap(jet), title('Classification Result');
end

function classedImg = classifyImage(model, image)
  imgSize = size(image);
  FeatureVects = reshape(image, [imgSize(1)*imgSize(2),imgSize(3)]);
  dimensionVectors = threeDSpace(model, FeatureVects);
  [maxValue,maxPos] = max(dimensionVectors,[],1);
  classedImg = uint8(reshape(maxPOS,size(1:2)));
end
