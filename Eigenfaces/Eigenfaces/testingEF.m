function testingEF(testImage,images,H,W,M,m,U,omega)
    %load the test image to be recognized
    testIm=imread(testImage);
        
    im=reshape(testIm,H*W,1);
    imtest=double(im);
    
    imd=imtest-m;
    
    % projection of the test face on the eigenfaces
    om=U'*imd;

    d=repmat(om,1,M)-omega;
    
    dist=zeros(M,1);
     
    % find the distance from all training faces
    for i=1:M
        dist(i,1)=norm(d(:,i));
    end

    % index corresponding to the minimum of the distances
    index=IndexOfMinimum(dist);
    
    % display the results
    subplot(1,2,1)
    imshow(testImage)
    title('Test face')
    subplot(1,2,2)
    imshow(uint8(images(:,:,index)))
    title('Recognized face')
end