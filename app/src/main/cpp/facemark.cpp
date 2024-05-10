#include "facemark.h"

bool FacemarkLBFImpl::fitImpl(const Mat image, std::vector<Point2f>& landmarks){
    if (landmarks.size()>0)
        landmarks.clear();

    if (!isModelTrained) {
        CV_Error(Error::StsBadArg, "The LBF model is not trained yet. Please provide a trained model.");
    }

    Mat img;
    if(image.channels()>1){
        cvtColor(image,img,COLOR_BGR2GRAY);
    }else{
        img = image;
    }

    Rect box;
    if (params.detectROI.width>0){
        box = params.detectROI;
    }else{
        std::vector<Rect> rects;

        if (!getFaces(img, rects)) return 0;
        if (rects.empty())  return 0; //failed to get face
        box = rects[0];
    }

    double min_x, min_y, max_x, max_y;
    min_x = std::max(0., (double)box.x - box.width / 2);
    max_x = std::min(img.cols - 1., (double)box.x+box.width + box.width / 2);
    min_y = std::max(0., (double)box.y - box.height / 2);
    max_y = std::min(img.rows - 1., (double)box.y + box.height + box.height / 2);

    double w = max_x - min_x;
    double h = max_y - min_y;

    BBox bbox(box.x - min_x, box.y - min_y, box.width, box.height);
    Mat crop = img(Rect((int)min_x, (int)min_y, (int)w, (int)h)).clone();
    Mat shape = regressor.predict(crop, bbox);

    if(params.detectROI.width>0){
        landmarks = Mat(shape.reshape(2)+Scalar(min_x, min_y));
        params.detectROI.width = -1;
    }else{
        landmarks = Mat(shape.reshape(2)+Scalar(min_x, min_y));
    }

    return 1;
}