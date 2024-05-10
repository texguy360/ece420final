package com.ece420.finalproj;

import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.Manifest;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
//import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.face.*;
import org.opencv.core.*;
import org.opencv.objdetect.CascadeClassifier;

import java.io.*;
import android.content.Context;
import android.util.Log;
import java.util.List;
import java.util.ArrayList;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    static {
       System.loadLibrary("finalproj");
    }

    private static final String TAG = "MainActivity";

    // UI Variables
    private Button controlButton;
    private SeekBar colorSeekbar;
    private SeekBar widthSeekbar;
    private SeekBar heightSeekbar;
    private TextView widthTextview;
    private TextView heightTextview;

    // Declare OpenCV based camera view base
    private CameraBridgeViewBase mOpenCvCameraView;
    // Camera size
    private int myWidth;
    private int myHeight;

    // Mat to store RGBA and Grayscale camera preview frame
    private Mat mRgba;
    private Mat mGray;

    // KCF Tracker variables
//    private TrackerKCF myTracker;
    String modelPath; // path of model used
    private CascadeClassifier faceCascade;
    private Facemark facemark;
    private Rect2d myROI = new Rect2d(0,0,0,0);
    private int myROIWidth = 70;
    private int myROIHeight = 70;
    private Scalar myROIColor = new Scalar(0,0,0);
    private int tracking_flag = -1;

    private String absFM;
    private String absFaceModel;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        super.setRequestedOrientation (ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        // Request User Permission on Camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 1);}

        // OpenCV Loader and Avoid using OpenCV Manager
        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

        // Setup color seek bar
        colorSeekbar = (SeekBar) findViewById(R.id.colorSeekBar);
        colorSeekbar.setProgress(50);
        setColor(50);
        colorSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener()
        {
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser)
            {
                setColor(progress);
            }
            public void onStartTrackingTouch(SeekBar seekBar) {}
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Setup width seek bar
        widthTextview = (TextView) findViewById(R.id.widthTextView);
        widthSeekbar = (SeekBar) findViewById(R.id.widthSeekBar);
        widthSeekbar.setProgress(myROIWidth - 20);
        widthSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener()
        {
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser)
            {
                // Only allow modification when not tracking
                if(tracking_flag == -1) {
                    myROIWidth = progress + 20;
                }
            }
            public void onStartTrackingTouch(SeekBar seekBar) {}
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Setup width seek bar
        heightTextview = (TextView) findViewById(R.id.heightTextView);
        heightSeekbar = (SeekBar) findViewById(R.id.heightSeekBar);
        heightSeekbar.setProgress(myROIHeight - 20);
        heightSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener()
        {
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser)
            {
                // Only allow modification when not tracking
                if(tracking_flag == -1) {
                    myROIHeight = progress + 20;
                }
            }
            public void onStartTrackingTouch(SeekBar seekBar) {}
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Setup control button
        controlButton = (Button)findViewById((R.id.controlButton));
        controlButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (tracking_flag == -1) {
                    // Modify UI
                    controlButton.setText("STOP");
                    // Modify tracking flag
                    tracking_flag = 0;
                }
                else if(tracking_flag == 1){
                    // Modify UI
                    controlButton.setText("START");
                    // Tear down myTracker
                    facemark.clear();
                    // Modify tracking flag
                    tracking_flag = -1;
                }
            }
        });

        // Setup OpenCV Camera View
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_camera_preview);
        // Use main camera with 0 or front camera with 1
        mOpenCvCameraView.setCameraIndex(0);
        // Force camera resolution, ignored since OpenCV automatically select best ones
        // mOpenCvCameraView.setMaxFrameSize(1280, 720);
        mOpenCvCameraView.setCvCameraViewListener(this);
        InputStream fmInputStream = getApplicationContext().getResources().openRawResource(R.raw.lbfmodel);

        // Create a private directory
        File faceDir = getDir("facelib", Context.MODE_PRIVATE);

        File fmModel = new File(faceDir, "lbfmodel.yaml");
        absFM = fmModel.getAbsolutePath();

        if (!fmModel.exists()) {
            try {
                fileCopy(fmModel, fmInputStream);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            assert fmModel.exists() : "Out of space! Failed to create " + "lbfmodel.yaml";
            Log.i(TAG, "Done created: " + "lbfmodel.yaml");
        }
    }

    private void fileCopy(File mFile, InputStream mIs) throws IOException {
        // Output stream
        FileOutputStream mOs = new FileOutputStream(mFile);

        byte[] buffer = new byte[4096];
        int byteRead = mIs.read(buffer);
        while (byteRead != -1) {
            mOs.write(buffer, 0, byteRead);
            byteRead = mIs.read(buffer);
        }

        mIs.close();
        mOs.close();
    }


    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    private Facemark createFacemarkASM() {
        return Face.createFacemarkLBF();
    }

    public void setColor(int value) {
        double a=(1-(double)value/100)/0.2;
        int X=(int)Math.floor(a);
        int Y=(int)Math.floor(255*(a-X));
        double newColor[] = {0,0,0};
        switch(X)
        {
            case 0:
                // r=255;g=Y;b=0;
                newColor[0] = 255;
                newColor[1] = Y;
                break;
            case 1:
                // r=255-Y;g=255;b=0
                newColor[0] = 255-Y;
                newColor[1] = 255;
                break;
            case 2:
                // r=0;g=255;b=Y
                newColor[1] = 255;
                newColor[2] = Y;
                break;
            case 3:
                // r=0;g=255-Y;b=255
                newColor[1] = 255-Y;
                newColor[2] = 255;
                break;
            case 4:
                // r=Y;g=0;b=255
                newColor[0] = Y;
                newColor[2] = 255;
                break;
            case 5:
                // r=255;g=0;b=255
                newColor[0] = 255;
                newColor[2] = 255;
                break;
        }
        myROIColor.set(newColor);
        return;
    }

    // OpenCV Camera Functionality Code
    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        myWidth = width;
        myHeight = height;
        myROI = new Rect2d(myWidth / 2 - myROIWidth / 2,
                            myHeight / 2 - myROIHeight / 2,
                            myROIWidth,
                            myROIHeight);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // Timer
        long start = Core.getTickCount();
        // Grab camera frame in rgba and grayscale format
        mRgba = inputFrame.rgba();
        // Grab camera frame in gray format
        mGray = inputFrame.gray();

        // Action based on tracking flag
        if(tracking_flag == -1){
            // Update myROI to keep the window to the center
            myROI.x = myWidth / 2 - myROIWidth / 2;
            myROI.y = myHeight / 2 - myROIHeight / 2;
            myROI.width = myROIWidth;
            myROI.height = myROIHeight;
        }
        else if(tracking_flag == 0){
            facemark = createFacemarkASM();
            facemark.loadModel(absFM);
            tracking_flag=1;
        }
        else{

            // Initialize a list to store the detected landmarks
            List<MatOfPoint2f> landmarks = new ArrayList<>();

            // Create a Rect object representing the entire screen
            Rect screenRect = new Rect((int)((myWidth - myROIWidth) / 2), (int)((myHeight - myROIHeight) / 2), (int)(myROIWidth), (int)(myROIHeight));

            Rect[] rectsArray = new Rect[] { screenRect };

            MatOfRect faces = new MatOfRect(rectsArray);

            // Detect landmarks
            boolean success = facemark.fit(mRgba, faces, landmarks);

            for (int i = 0; i < landmarks.size(); i++) {
                // Get the landmarks for the current face
                MatOfPoint2f landmarksVector = landmarks.get(i);

                // Convert the landmarks to an array of points
                Point[] points = landmarksVector.toArray();

                // Draw each landmark point on the image
                for (int j = 48; j <= 67; j++) {
                    Point point = points[j];
                    Imgproc.circle(mRgba, point, 3, new Scalar(0, 255, 0), -1); // Draw a green circle at the mouth landmark point
                }
            }

            long end = Core.getTickCount();
            double fps = Core.getTickFrequency()/(end-start);
            Imgproc.putText(mRgba, "FPS@"+ (int) fps, new Point(60, 70),
                    Core.FONT_HERSHEY_SIMPLEX, 3, new Scalar(0, 0, 255), 2);
        }

        // Draw a rectangle on to the current frame
        Imgproc.rectangle(mRgba,
                          new Point(myROI.x, myROI.y),
                          new Point(myROI.x + myROI.width, myROI.y + myROI.height),
                          myROIColor,
                4);

        // Returned frame will be displayed on the screen
        return mRgba;
    }
}