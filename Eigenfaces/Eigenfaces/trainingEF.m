function [images,H,W,M,m,U,omega]=trainingEF(trainingFolder)
    % find the training files
    pgmFiles=dir(sprintf('%s/*.pgm',trainingFolder));

    im=imread(pgmFiles(1).name); % read an image to determine the Height & Width

    H=size(im,1); %Height of the images
    W=size(im,2); %Width of the images
    M=size(pgmFiles,1); %Number of images in the training set

    images=zeros(H,W,M);
    vec=zeros(H*W,M);

    % load the training images
    for i=1:M
        images(:,:,i)=imread(pgmFiles(i).name);
        vec(:,i)=reshape(images(:,:,i),H*W,1);
    end
        
    % mean face
    m=sum(vec,2)/M;

    % face space
    A=vec-repmat(m,1,M);

    L=A'*A;
    [V,lambda]=eig(L);

    % eigenvector of the covariance matrix of A. These are the eigenfaces
    U=A*V;

    % projection of each vector in the face space A on the eigenfaces
    omega=U'*A;

end

