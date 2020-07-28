# FACE RECOGNITION

  PART 1: Face Detection - Use pre-trained Cascade will be faster than pre-trained Face Recognition - which is built from CNN but pre-trained Face Recognition detected better than Cascade. Another way we can use Dual Shot Face Detector and RetinaFace, it will give an amazing result. Time for Face Recognition is about 0.2s per image, and DSFD is about 0.9s per image when I run in Google Colab. 
  
  PART 2: Face Recognition
   - nn4.small2.v1.t7 is trained with a combination of the two largest (of August 2015) publicly-available face recognition datasets based on names: FaceScrub and CASIA-WebFace.
 
   + On Khanh Dataset, it's too hard to recognition face because images in this dataset are different lighting, angle, ... and it's small. It made CNN model overfitting. In this case, using the pre-trained model to bring better results. 
   
   + On YALE Dataset, it's easy to recognition face because it is preprocessed. We get nice results on this dataset. However it's still a small dataset, we must using data augmentation to increase accuracy.
  
   + Loss function in this project is Triplet loss: 
        
        + Mathematically, it is defined as: L=max(d(a,p)−d(a,n)+margin,0).

        + We minimize this loss, which pushes d(a,p) to 0 and d(a,n) to be greater than d(a,p)+margin. This means that, after the training, the positive examples will be closer to the anchor while the negative examples will be farther from it. The image below shows the effect of minimizing the loss.

![alt Triplet Loss Image](https://github.com/congdaoduy298/Face-Detection/blob/master/images/triplet_loss.png)

# FACE RECOGNITION VIDEO
  
  Just by a small dataset, about 30 images per person, we can recognize faces with a good result. Encoding faces by face_encodings of face_recognition library. 
 The face recognition model, which I made before it worked so badly when I trained in this dataset because of the dataset's small.  
  
    
# REFERENCES
  
  Mô hình Facenet trong face recognition  https://phamdinhkhanh.github.io/2020/03/12/faceNetAlgorithm.html
  
  Thực hành Training Facenet  https://phamdinhkhanh.github.io/2020/03/21/faceNet.html
  
  Models and Accuracies https://cmusatyalab.github.io/openface/models-and-accuracies/
    
  Triplet Loss and Siamese Neural Networks  https://medium.com/@enoshshr/triplet-loss-and-siamese-neural-networks-5d363fdeba9b
  
  Facial Recognition with Python and the face_recognition library https://www.youtube.com/watch?v=535acCxjHCI&t=861s
