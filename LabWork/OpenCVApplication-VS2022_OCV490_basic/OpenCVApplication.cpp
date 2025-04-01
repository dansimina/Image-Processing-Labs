// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <math.h>
#include <unordered_set>
#include <random>

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

// Lab 1

// Implement a function which changes the gray levels of an image by an additive factor.
void changeGrayLevelUsingAdditiveFactor() {
	unsigned int factor = 0;
	printf("\nFactor: ");
	scanf("%d", &factor);

	Mat img = imread("Images/saturn.bmp", 0);

	imshow("original", img);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int val = img.at<unsigned char>(i, j) + factor;
			img.at<unsigned char>(i, j) = val > 255 ? 255 : val;
		}
	}

	imshow("new_image", img);
	waitKey(0);
}

// Implement a function which changes the gray levels of an image by a multiplicative factor.
void changeGrayLevelUsingMultiplicativeFactor() {
	unsigned int factor = 0;
	printf("\nFactor: ");
	scanf("%d", &factor);

	Mat img = imread("Images/saturn.bmp", 0);

	imshow("original", img);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int val = img.at<unsigned char>(i, j) * factor;
			img.at<unsigned char>(i, j) = val > 255 ? 255 : val;
		}
	}

	imshow("new_image", img);
	imwrite("MyImages/imgEx4.bmp", img);
	waitKey(0);
}

//Create a color image of dimension 256 x 256. Divide it into 4 squares and color the squares from top to bottom, left to right as : white, red, green, yellow.
void createImageWithFourColors() {
	Mat img(256, 256, CV_8UC3);

	int mid = 128;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b color{ 0, 0, 0 };

			if (i < mid && j < mid) {
				color = { 255, 255, 255 };
			}
			else if (i < mid && j >= mid) {
				color = { 0, 0, 255 };
			}
			else if (i >= mid && j < mid) {
				color = { 0, 255, 0 };
			}
			else if (i >= mid && j >= mid) {
				color = { 0, 255, 255 };
			}

			img.at<Vec3b>(i, j) = color;
		}
	}

	imshow("image", img);
	waitKey(0);
}

//Create a 3x3 float matrix, determine its inverse and print it. 

void computeMatrixInverse() {
	printf("\Matrix 3x3: ");
	float vals[9]{};

	for (int i = 0; i < 9; i++) {
		scanf("%f", &vals[i]);
	}
	
	Mat M(3, 3, CV_32FC1, vals);

	M = M.inv();

	printf("\n");
	for (int i = 0; i < M.rows; i++) {
		for (int j = 0; j < M.cols; j++) {
			printf("%f ", M.at<float>(i, j));
		}
		printf("\n");
	}

	char c;
	scanf("%c", &c);
	scanf("%c", &c);
}

// Lab 2
//1.Create a function that will copy the R, G and B channels of a color, RGB24 image(CV_8UC3 type) into three matrices of type CV_8UC1(grayscale images).Display these matrices in three distinct windows.

void imageTo3GrayscaleImages() {
	Mat img = imread("Images/flowers_24bits.bmp",1);

	Mat imgR(img.rows, img.cols, CV_8UC3);
	Mat imgG(img.rows, img.cols, CV_8UC3);
	Mat imgB(img.rows, img.cols, CV_8UC3);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			imgR.at<Vec3b>(i, j) = { 0, 0, img.at<Vec3b>(i, j)[2] };
			imgG.at<Vec3b>(i, j) = { 0, img.at<Vec3b>(i, j)[1], 0 };
			imgB.at<Vec3b>(i, j) = { img.at<Vec3b>(i, j)[0], 0, 0 };
		}
	}

	imshow("img", img);
	imshow("red", imgR);
	imshow("green", imgG);
	imshow("blue", imgB);

	waitKey(0);
}

// Create a function that will convert a color RGB24 image (CV_8UC3 type) to a grayscale image(CV_8UC1), and display the result image in a destination window.
void colorImageToGrayscaleImage() {
	Mat img = imread("Images/flowers_24bits.bmp", 1);

	Mat newImg(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			newImg.at<unsigned char>(i, j) = (img.at<Vec3b>(i, j)[0] + img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j)[2]) / 3.0;
		}
	}

	imshow("InitialImage", img);
	imshow("GrayscaleImage", newImg);
	waitKey(0);
}

//Create a function for converting from grayscale to black and white (binary), using (2.2). Read the threshold from the console. Test the operation on multiple images, and using multiple thresholds.
void grayscaleToBlackAndWhite() {
	Mat img = imread("Images/eight.bmp", 0);
	Mat binaryImg(img.rows, img.cols, CV_8UC1);

	int threshold = 0;
	printf("Threshold: ");
	scanf("%d", &threshold);

	threshold %= 256;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<unsigned char>(i, j) > threshold) {
				binaryImg.at<unsigned char>(i, j) = 255;
			}
			else {
				binaryImg.at<unsigned char>(i, j) = 0;
			}
		}
	}

	imshow("InitialImage", img);
	imshow("BinaryImage", binaryImg);

	waitKey(0);
}

//Create a function that will compute the H, S and V values from the R, G, B channels of an image, using the equations from 2.6. Store each value (H, S, V) in a CV_8UC1 matrix. Display these matrices in distinct windows. Check the correctness of your implementation using the example below.
void computeHSV(Mat img, Mat& imgH, Mat& imgS, Mat& imgV) {
	imgH = Mat(img.rows, img.cols, CV_8UC1);
	imgS = Mat(img.rows, img.cols, CV_8UC1);
	imgV = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			float r = img.at<Vec3b>(i, j)[2] / 255.0;
			float g = img.at<Vec3b>(i, j)[1] / 255.0;
			float b = img.at<Vec3b>(i, j)[0] / 255.0;

			float M = max(r, max(g, b));
			float m = min(r, min(g, b));
			float C = M - m;

			float H = 0;
			float S = 0;
			float V = M;

			if (V != 0) {
				S = C / V;
			}
			else {
				S = 0;
			}

			if (C != 0) {
				if (M == r)
					H = 60 * (g - b) / C;
				else if (M == g) {
					H = 120 + 60 * (b - r) / C;
				}
				else if (M == b) {
					H = 240 + 60 * (r - g) / C;
				}
			}
			else {
				H = 0;
			}

			if (H < 0) {
				H = H + 360;
			}

			unsigned char Hnorm = H * 255 / 360;
			unsigned char Snorm = S * 255;
			unsigned char Vnorm = V * 255;

			imgH.at<unsigned char>(i, j) = Hnorm;
			imgS.at<unsigned char>(i, j) = Snorm;
			imgV.at<unsigned char>(i, j) = Vnorm;
		}
	}
}

void displayHSV() {
	Mat img = imread("Images/Lena_24bits.bmp", 1);
	Mat imgH;
	Mat imgS;
	Mat imgV;

	computeHSV(img, imgH, imgS, imgV);

	imshow("img", img);
	imshow("H", imgH);
	imshow("S", imgS);
	imshow("V", imgV);

	waitKey(0);
}

// Lab 3
//Compute the histogram for a given grayscale image (in an array of integers having dimension 256).
// Compute the histogram for a given number of bins m ≤ 256.
int* computeHistogram(Mat img, int m) {
	int* histogram = new int[256]();

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			histogram[img.at<unsigned char>(i, j)]++;
		}
	}

	if (m < 256) {
		int* old = histogram;
		int quotinet = 256 / m;
		int remainder = 256 % m;
		histogram = new int[256]();

		for (int i = 0; i < 256; i += quotinet) {
			for (int j = i; j < i + quotinet; j++) {
				histogram[i] += old[j];
			}
			if (remainder > 0) {
				histogram[i] += old[i + quotinet];
				i++;
				remainder--;
			}
		}

		delete(old);
	}

	return histogram;
}

// Compute the PDF(in an array of floats of dimension 256).
float* probabilityDensityFunction(Mat img, int m) {
	int* histogram = computeHistogram(img, m);

	int M = img.rows * img.cols;

	float* pdf = new float[256]();

	for (int i = 0; i < 256; i++) {
		pdf[i] = (float)histogram[i] / M;
	}

	delete(histogram);

	return pdf;
}

void displayPDF() {
	Mat img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	float* res = probabilityDensityFunction(img, 256);
	for (int i = 0; i < 256; i++) {
		printf("%f ", res[i]);
	}
	int c;
	scanf("%c", &c);
	scanf("%c", &c);
	delete(res);
}

// Display the computed histogram
void displayHistogram(int* (*func)(Mat img, int len), int len) {
	Mat img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);

	int* histogram = func(img, len);

	showHistogram("histogram", histogram, 256, 200);
	waitKey(0);

	delete(histogram);
}

// Implement the multilevel thresholding algorithm from section.
Mat multilevelThresholdingAlgorithm(Mat img) {
	Mat newImg;
	img.copyTo(newImg);

	float* normalizedHistogram = probabilityDensityFunction(newImg, 256);
	int WH = 5;
	int windowWidth = 2 * WH + 1;

	std::deque<int> q;

	float TH = 0.0003;

	for (int i = WH; i <= 255 - WH; i++) {
		float avg = 0.0;
		bool ok = true;
		for (int j = i - WH; j <= i + WH; j++) {
			avg += normalizedHistogram[j];
			if (j != i && normalizedHistogram[i] < normalizedHistogram[j]) {
				ok = false;
			}
		}
		avg /= windowWidth;

		if (normalizedHistogram[i] > avg + TH && ok) {
			q.push_back(i);
		}
	}

	q.push_front(0);
	q.push_back(255);

	for (int i = 0; i < newImg.rows; i++) {
		for (int j = 0; j < newImg.cols; j++) {
			int newPixel = 0;
			int dist = 256;

			int oldPixel = newImg.at<unsigned char>(i, j);

			for (int k = 0; k < q.size(); k++) {
				if (abs(oldPixel - q[k]) < dist) {
					dist = abs(oldPixel - q[k]);
					newPixel = q[k];
				}
			}

			newImg.at<unsigned char>(i, j) = newPixel;
		}
	}

	return newImg;
}

void displayImgAfterMultilevelThresholdingAlgorithm() {
	Mat img = imread("Images/saturn.bmp", IMREAD_GRAYSCALE);
	Mat newImg;

	newImg = multilevelThresholdingAlgorithm(img);

	imshow("img", img);
	imshow("newImg", newImg);
	waitKey(0);
}


// Enhance the multilevel thresholding algorithm using the Floyd - Steinberg dithering from section 3.4.
int clamp(int value, int min, int max) {
	if (value < min) return min;
	if (value > max) return max;
	return value;
}

Mat multilevelThresholdingAlgorithmWithFloydSteinberg(Mat img) {
	Mat newImg;
	img.copyTo(newImg);

	float* normalizedHistogram = probabilityDensityFunction(newImg, 256);
	int WH = 5;
	int windowWidth = 2 * WH + 1;

	std::deque<int> q;

	float TH = 0.0003;

	for (int i = WH; i <= 255 - WH; i++) {
		float avg = 0.0;
		bool ok = true;
		for (int j = i - WH; j <= i + WH; j++) {
			avg += normalizedHistogram[j];
			if (j != i && normalizedHistogram[i] < normalizedHistogram[j]) {
				ok = false;
			}
		}
		avg /= windowWidth;

		if (normalizedHistogram[i] > avg + TH && ok) {
			q.push_back(i);
		}
	}

	q.push_front(0);
	q.push_back(255);

	for (int i = 0; i < newImg.rows; i++) {
		for (int j = 0; j < newImg.cols; j++) {
			int newPixel = 0;
			int dist = 256;

			int oldPixel = newImg.at<unsigned char>(i, j);

			for (int k = 0; k < q.size(); k++) {
				if (abs(oldPixel - q[k]) < dist) {
					dist = abs(oldPixel - q[k]);
					newPixel = q[k];
				}
			}

			newImg.at<unsigned char>(i, j) = newPixel;

			int error = oldPixel - newPixel;
			if (j + 1 < newImg.cols) {
				newImg.at<unsigned char>(i, j + 1) = clamp(newImg.at<unsigned char>(i, j + 1) + (int)(7 * error / 16.0), 0, 255);
			}
			if (i + 1 < newImg.rows && j - 1 >= 0) {  
				newImg.at<unsigned char>(i + 1, j - 1) = clamp(newImg.at<unsigned char>(i + 1, j - 1) + (int)(3 * error / 16.0), 0, 255);
			}
			if (i + 1 < newImg.rows) {
				newImg.at<unsigned char>(i + 1, j) = clamp(newImg.at<unsigned char>(i + 1, j) + (int)(5 * error / 16.0), 0, 255);
			}
			if (i + 1 < newImg.rows && j + 1 < newImg.cols) {
				newImg.at<unsigned char>(i + 1, j + 1) = clamp(newImg.at<unsigned char>(i + 1, j + 1) + (int)(error / 16.0), 0, 255);
			}

		}
	}

	return newImg;
}

void displayImgAfterMultilevelThresholdingAlgorithmWithFloydSteinberg() {
	Mat img = imread("Images/saturn.bmp", IMREAD_GRAYSCALE);
	Mat newImg;

	newImg = multilevelThresholdingAlgorithmWithFloydSteinberg(img);

	imshow("img", img);
	imshow("newImg", newImg);
	waitKey(0);
}

//Perform multilevel thresholding on a color image by applying the procedure from 3.3 on
//the Hue channel from the HSV color - space representation of the image.Modify only the
//Hue values, keeping the S and V channels unchanged or setting them to their maximum
//possible value.Transform the result back to RGB color - space for viewing.
Mat multilevelThresholdingOnColorImage(Mat img) {
	Mat imgH;
	Mat imgS;
	Mat imgV;

	computeHSV(img, imgH, imgS, imgV);

	imgH = multilevelThresholdingAlgorithmWithFloydSteinberg(imgH);

	Mat newImg(img.rows, img.cols, CV_8UC3);

	for (int i = 0; i < newImg.rows; i++) {
		for (int j = 0; j < newImg.cols; j++) {
			float H = imgH.at<unsigned char>(i, j) * 360.0 / 255.0;
			float S = imgS.at<unsigned char>(i, j) / 255.0;
			float V = imgV.at<unsigned char>(i, j) / 255.0;

			float C = V * S;
			float X = C * (1 - abs((int)(H / 60) % 2 - 1));
			float m = V - C;

			Vec3b pixel;

			if (H >= 0 && H < 60) {
				pixel = Vec3b(
					clamp((0 + m) * 255, 0, 255),
					clamp((X + m) * 255, 0, 255),
					clamp((C + m) * 255, 0, 255)
				);
			}
			else if (H >= 60 && H < 120) {
				pixel = Vec3b(
					clamp((0 + m) * 255, 0, 255),
					clamp((C + m) * 255, 0, 255),
					clamp((X + m) * 255, 0, 255)
				);
			}
			else if (H >= 120 && H < 180) {
				pixel = Vec3b(
					clamp((X + m) * 255, 0, 255),
					clamp((C + m) * 255, 0, 255),
					clamp((0 + m) * 255, 0, 255)
				);
			}
			else if (H >= 180 && H < 240) {
				pixel = Vec3b(
					clamp((C + m) * 255, 0, 255),
					clamp((X + m) * 255, 0, 255),
					clamp((0 + m) * 255, 0, 255)
				);
			}
			else if (H >= 240 && H < 300) {
				pixel = Vec3b(
					clamp((C + m) * 255, 0, 255),
					clamp((0 + m) * 255, 0, 255),
					clamp((X + m) * 255, 0, 255)
				);
			}
			else if (H >= 300 && H < 360) {
				pixel = Vec3b(
					clamp((X + m) * 255, 0, 255),
					clamp((0 + m) * 255, 0, 255),
					clamp((C + m) * 255, 0, 255)
				);
			}
			else {
				pixel = Vec3b(0, 0, 0);
			}

			newImg.at<Vec3b>(i, j) = pixel;
		}
	}

	return newImg;
}

void displayMultilevelThresholdingOnColorImage() {
	Mat img = imread("Images/Lena_24bits.bmp", 1);

	Mat newImg = multilevelThresholdingOnColorImage(img);

	imshow("img", img);
	imshow("new img", newImg);
	waitKey(0);
}

// Lab 4
//For a specific object in a labeled image selected by a mouse click, compute the object’s area, center of mass, axis of elongation, perimeter, thinness ratio and aspect ratio.

int computeArea(Mat img, int x, int y) {
	int area = 0;
	
	Vec3b targetColor = img.at<Vec3b>(y, x);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j)[0] == targetColor[0] &&
				img.at<Vec3b>(i, j)[1] == targetColor[1] &&
				img.at<Vec3b>(i, j)[2] == targetColor[2]) {
				area++;
			}
		}
	}

	return area;
}

std::pair<int, int> computeCenterOfMass(Mat img, int x, int y) {
	int area = computeArea(img, x, y);

	int r = 0;
	int c = 0;

	Vec3b targetColor = img.at<Vec3b>(y, x);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j)[0] == targetColor[0] &&
				img.at<Vec3b>(i, j)[1] == targetColor[1] &&
				img.at<Vec3b>(i, j)[2] == targetColor[2]) {
				r += i;
				c += j;
			}
		}
	}

	return { r / area, c / area };
}

double computeAxisOfElongation(Mat img, int x, int y) {
	Vec3b targetColor = img.at<Vec3b>(y, x);
	std::pair<int, int> centerOfMass = computeCenterOfMass(img, x, y);

	int t1 = 0, t2 = 0, t3 = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j)[0] == targetColor[0] &&
				img.at<Vec3b>(i, j)[1] == targetColor[1] &&
				img.at<Vec3b>(i, j)[2] == targetColor[2]) {
				
				t1 += (i - centerOfMass.first) * (j - centerOfMass.second);
				t2 += (j - centerOfMass.second) * (j - centerOfMass.second);
				t3 += (i - centerOfMass.first) * (i - centerOfMass.first);
			}
		}
	}

	t1 = (t1 << 1);

	return atan2(t1, (t2 - t3)) * 0.5;
}

int computePerimeter(Mat img, int x, int y) {
	int perimeter = 0;

	Vec3b targetColor = img.at<Vec3b>(y, x);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j) == targetColor && 
				((i < img.rows && img.at<Vec3b>(i + 1, j) != targetColor) ||
					(i >= 0 && img.at<Vec3b>(i - 1, j) != targetColor) ||
					(j < img.cols && img.at<Vec3b>(i, j + 1) != targetColor) ||
					(j >= 0 && img.at<Vec3b>(i, j - 1) != targetColor)
					)) {
				
				perimeter++;
			}
		}
	}

	return perimeter * (PI / 4);
}

double computeThinnessRatio(Mat img, int x, int y) {
	int area = computeArea(img, x, y);
	int perimeter = computePerimeter(img, x, y);

	return 4.0 * PI * ((double) area / (perimeter * perimeter));
}

double computeAspectRatio(Mat img, int x, int y) {
	int cMin = img.cols;
	int rMin = img.rows;
	int cMax = 0;
	int rMax = 0;

	Vec3b targetColor = img.at<Vec3b>(y, x);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j) == targetColor) {
				cMin = min(cMin, j);
				rMin = min(rMin, i);
				cMax = max(cMax, j);
				rMax = max(rMax, i);
			}
		}
	}

	return (double)(cMax - cMin + 1) / (rMax - rMin + 1);
}

void drawSelectedObject(Mat img, int x, int y) {
	Mat newImg(img.rows, img.cols, CV_8UC3);

	Vec3b targetColor = img.at<Vec3b>(y, x);

	std::pair<int, int> centerOfMass = computeCenterOfMass(img, x, y);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j) == targetColor &&
				((i < img.rows && img.at<Vec3b>(i + 1, j) != targetColor) ||
					(i >= 0 && img.at<Vec3b>(i - 1, j) != targetColor) ||
					(j < img.cols && img.at<Vec3b>(i, j + 1) != targetColor) ||
					(j >= 0 && img.at<Vec3b>(i, j - 1) != targetColor)
					)) {

				newImg.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
			}

			if (abs(i - centerOfMass.first) + abs(j - centerOfMass.second) < 5) {
				newImg.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
			}
		}
	}

	double angle = computeAxisOfElongation(img, x, y);
	int length = 100;

	Point p1, p2;
	p1.x = centerOfMass.second - length * cos(angle);
	p1.y = centerOfMass.first - length * sin(angle);
	p2.x = centerOfMass.second + length * cos(angle);
	p2.y = centerOfMass.first + length * sin(angle);
	line(newImg, p1, p2, Scalar(255, 0, 0), 2);

	imshow("Img1", newImg);
}

void computeProjections(Mat img, int x, int y) {
	Mat projectionOnY(img.rows, img.cols, CV_8UC3);
	Mat projectionOnX(img.rows, img.cols, CV_8UC3);

	std::vector<int> r(img.rows, 0);
	std::vector<int> c(img.cols, 0);

	Vec3b targetColor = img.at<Vec3b>(y, x);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j) == targetColor) {
				r[i]++;
				c[j]++;
			}
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < r[i]; j++) {
			projectionOnY.at<Vec3b>(i, j) = Vec3b(0, 255, 0);

		}
	}

	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < c[i]; j++) {
			projectionOnX.at<Vec3b>(img.rows - j - 1, i) = Vec3b(0, 255, 0);
		}
	}

	imshow("Projection on X", projectionOnX);
	imshow("Projection on Y", projectionOnY);
}

void myCallBackFuncObjectProperties(int event, int x, int y, int flags, void* param)
{
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		printf("Area: %d\n", computeArea((*src), x, y));

		std::pair<int, int> centerOfMass = computeCenterOfMass((*src), x, y);
		printf("Center of mass: %d %d\n", centerOfMass.first, centerOfMass.second);

		printf("The axis of elongatio: %lf\n", computeAxisOfElongation((*src), x, y));

		printf("Perimeter: %d\n", computePerimeter((*src), x, y));

		printf("The thinness ratio: %lf\n", computeThinnessRatio((*src), x, y));

		printf("The aspect ratio: %lf\n", computeAspectRatio((*src), x, y));

		drawSelectedObject((*src), x, y);

		computeProjections((*src), x, y);
	}
}

void mouseClickObjectProperties()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", myCallBackFuncObjectProperties, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

// Create a new processing function which takes as input a labeled image and keeps in the output image only the objects that :
//	a.have their area < TH_area
//	b.have a specific orientation phi, where phi_LOW < phi < phi_HIGH where TH_area, phi_LOW, phi_HIGH are given by the user.

int colorToInt(const Vec3b& color) {
	return (color[0] << 16) | (color[1] << 8) | (color[2]);
}

Mat filterObjects(Mat img, int th, float phi_low, float phi_high) {
	std::unordered_map<int, int> area, t1, t2, t3;
	std::unordered_map<int, float> r, c;
	std::unordered_map<int, float> phi;

	Mat newImg(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b color = img.at<Vec3b>(i, j);
			int key = colorToInt(color);
			if (key) {
				area[key]++;
				r[key] += i;
				c[key] += j;
			}
		}
	}

	for (auto& i : area) {
		r[i.first] /= (float)i.second;
		c[i.first] /= (float)i.second;
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b color = img.at<Vec3b>(i, j);
			int key = colorToInt(color);
			if (key) {
				t1[key] += (i - r[key]) * (j - c[key]);
				t2[key] += (j - c[key]) * (j - c[key]);
				t3[key] += (i - r[key]) * (i - r[key]);
			}
		}
	}

	for (auto& i : area) {
		phi[i.first] = atan2(2.0f * t1[i.first], (t2[i.first] - t3[i.first])) * 0.5f;
		if (phi[i.first] < 0) phi[i.first] += CV_PI;
	}

	int length = 200;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b color = img.at<Vec3b>(i, j);
			int key = colorToInt(color);
			if (key && area[key] < th && phi[key] >= phi_low && phi[key] <= phi_high) {
				newImg.at<Vec3b>(i, j) = color;
			}
		}
	}

	for (auto& i : phi) {
		int key = i.first;
		if (area[key] < th && phi[key] >= phi_low && phi[key] <= phi_high) {
			Point p1, p2;
			p1.x = c[key] - length * cos(phi[key]);
			p1.y = r[key] - length * sin(phi[key]);
			p2.x = c[key] + length * cos(phi[key]);
			p2.y = r[key] + length * sin(phi[key]);
			line(newImg, p1, p2, Scalar(0, 0, 0), 2);
		}
	}

	return newImg;
}

void displayFilteredObjects() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname);

		int th;
		float phi_low, phi_high;

		printf("\nTh: ");
		scanf("%d", &th);

		printf("\nPhi_low: ");
		scanf("%f", &phi_low);

		printf("\nPhi_high: ");
		scanf("%f", &phi_high);

		Mat filteredImg = filterObjects(src, th, phi_low, phi_high);

		namedWindow("Filtered Image", WINDOW_AUTOSIZE);
		imshow("Filtered Image", filteredImg);

		waitKey(0);
	}
}

// Lab 5

//Implement the breadth first traversal component labeling algorithm(Algorithm 1).You
//should be able to easily switch between the neighborhood types of 4 and 8.
std::vector<std::vector<int>> bfsComponentLabeling(Mat img, int neighborhood) {
	std::vector<std::vector<int>> labels(img.rows, std::vector<int>(img.cols, 0));
	
	if (neighborhood != 4 && neighborhood != 8)
		return labels;

	int di[] = {-1, 0, 1, 0, -1, 1, 1, -1};
	int dj[] = {0, -1, 0, 1, -1, -1, 1, 1};

	int label = 0;
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<unsigned char>(i, j) == 0 && labels[i][j] == 0) {
				label++;

				std::queue<std::pair<int, int>> q;
				q.push({ i, j });

				while (!q.empty()) {
					auto p = q.front();
					q.pop();

					int ci = p.first;
					int cj = p.second;

					for (int k = 0; k < neighborhood; k++) {
						int ni = ci + di[k];
						int nj = cj + dj[k];

						if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols) {
							if (img.at<uchar>(ni, nj) == 0 && labels[ni][nj] == 0) {
								labels[ni][nj] = label;
								q.push({ ni, nj });
							}
						}
					}

				}
			}
		}
	}

	return labels;
}

//Implement a function which generates a color image from a label matrix by assigning a
//random color to each label.Display the results.

void generateColorImage() {
	Mat img = imread("Images/letters.bmp", IMREAD_GRAYSCALE);
	Mat newImg(img.rows, img.cols, CV_8UC3);

	std::vector<std::vector<int>> labels = bfsComponentLabeling(img, 4);

	std::vector<Vec3b> colors(1000, Vec3b(0, 0, 0));

	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);

	for (int i = 0; i < labels.size(); i++) {
		for (int j = 0; j < labels[i].size(); j++) {
			if (labels[i][j] != 0) {
				if (colors[labels[i][j]] == Vec3b(0, 0, 0)) {
					Vec3b newColor = Vec3b(d(gen), d(gen), d(gen));
					colors[labels[i][j]] = newColor;
				}
				
				newImg.at<Vec3b>(i, j) = colors[labels[i][j]];
			}
			else {
				newImg.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
		}
	}

	imshow("init img", img);
	imshow("new img", newImg);
	waitKey(0);
}

//Implement the two - pass component labeling algorithm.Display the intermediate results
//you get after the first pass over the image.Compare this to the final results and to the
//previous algorithm.
std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> twoPassLabeling(Mat img) {
	std::vector<std::vector<int>> labels(img.rows, std::vector<int>(img.cols, 0));

	int di[] = { 0, -1, -1, -1 };
	int dj[] = { -1, -1, 0, 1 };

	int label = 0;

	std::vector<std::vector<int>> edges(1000);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<unsigned char>(i, j) == 0 && labels[i][j] == 0) {
				std::vector<std::pair<int, int>> L;

				for (int k = 0; k < 4; k++) {
					int ni = i + di[k];
					int nj = j + dj[k];

					if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols) {
						if (labels[ni][nj] > 0) {
							L.push_back({ ni, nj });
						}
					}
				}

				if (L.size() == 0) {
					label++;
					labels[i][j] = label;
				}
				else {
					int x = labels[L[0].first][L[0].second];
					for (auto& y : L) {
						x = min(x, labels[y.first][y.second]);
					}

					labels[i][j] = x;

					for (auto& y : L) {
						if (labels[y.first][y.second] != x) {
							edges[labels[y.first][y.second]].push_back(x);
							edges[x].push_back(labels[y.first][y.second]);
						}
					}
 				}
			}
		}
	}

	std::vector<std::vector<int>> interLabels(img.rows, std::vector<int>(img.cols, 0));
	for (int i = 0; i < labels.size(); i++) {
		for (int j = 0; j < labels[i].size(); j++) {
			interLabels[i][j] = labels[i][j];
		}
	}

	int newLabel = 0;
	std::vector<int> newLabels(label + 1);
	for (int i = 1; i <= label; i++) {
		if (newLabels[i] == 0) {
			newLabel++;
			std::queue<int> Q;
			newLabels[i] = newLabel;
			Q.push(i);

			while (!Q.empty()) {
				int x = Q.front();
				Q.pop();

				for (auto y : edges[x]) {
					if (newLabels[y] == 0) {
						newLabels[y] = newLabel;
						Q.push(y);
					}
				}
			}
		}
	}

	for (int i = 0; i < labels.size(); i++) {
		for (int j = 0; j < labels[i].size(); j++) {
			labels[i][j] = newLabels[labels[i][j]];
		}
	}

	return { interLabels, labels };
}

void generateColorImageTwoPasses() {
	Mat img = imread("Images/shapes.bmp", IMREAD_GRAYSCALE);
	Mat newImg1(img.rows, img.cols, CV_8UC3);
	Mat newImg2(img.rows, img.cols, CV_8UC3);

	std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> result = twoPassLabeling(img);
	std::vector<std::vector<int>> interLabels = result.first;
	std::vector<std::vector<int>> labels = result.second;

	std::vector<Vec3b> colors1(1000, Vec3b(0, 0, 0));
	std::vector<Vec3b> colors2(1000, Vec3b(0, 0, 0));

	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);

	for (int i = 0; i < labels.size(); i++) {
		for (int j = 0; j < labels[i].size(); j++) {
			if (labels[i][j] != 0) {
				if (colors1[labels[i][j]] == Vec3b(0, 0, 0)) {
					Vec3b newColor = Vec3b(d(gen), d(gen), d(gen));
					colors1[labels[i][j]] = newColor;
				}

				newImg1.at<Vec3b>(i, j) = colors1[labels[i][j]];
			}
			else {
				newImg1.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}

			if (interLabels[i][j] != 0) {
				if (colors2[interLabels[i][j]] == Vec3b(0, 0, 0)) {
					Vec3b newColor = Vec3b(d(gen), d(gen), d(gen));
					colors2[interLabels[i][j]] = newColor;
				}
				newImg2.at<Vec3b>(i, j) = colors2[interLabels[i][j]];
			}
			else {
				newImg2.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
		}
	}

	imshow("init img", img);
	imshow("inter img", newImg2);
	imshow("new img", newImg1);
	waitKey(0);
}

// Lab 6
// Implement the border tracing algorithm and draw the object contour on an image having a single object.

bool isInside(int i, int j, int rows, int cols) {
	return i >= 0 && i < rows && j >= 0 && j < cols;
}

void borderTracingAlgorithm(Mat img) {
	Mat result;
	img.copyTo(result);

	int arrDirI[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int arrDirJ[] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	int dir = 7;
	std::pair<int, int> p1, p2, current, prev;

	bool found = false;
	for (int i = 0; i < img.rows && !found; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j) != Vec3b(255, 255, 255)) {
				p1 = { i, j };
				p2 = p1;
				found = true;
				break;
			}
		}
	}

	if (!found) return; 

	current = p1;
	prev = p1;

	int ok = true;
	int n = 0;

	while (!(current == p2 && prev == p1 && n >= 2)) {
		n++;
		dir = dir % 2 == 0 ? (dir + 7) % 8 : (dir + 6) % 8;
		while (!(isInside(current.first + arrDirI[dir], current.second + arrDirJ[dir], img.rows, img.cols) &&
			img.at<Vec3b>(current.first + arrDirI[dir], current.second + arrDirJ[dir]) != Vec3b(255, 255, 255))) {
			dir = (dir + 1) % 8;
		}

		prev = current;
		current = { current.first + arrDirI[dir], current.second + arrDirJ[dir] };

		if (ok) {
			p2 = current;
			ok = false;
		}

		Point center(current.second, current.first);
		Scalar color(0, 0, 255);
		int radius = 1;
		int thickness = -1;

		circle(result, center, radius, color, thickness);
	}

	imshow("img", img);
	imshow("res", result);
	waitKey(0);
}

void displayBorderTracingAlgorithm() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname);

		int th;
		float phi_low, phi_high;

		borderTracingAlgorithm(src);
	}
}


//Starting from the border tracing algorithm write the algorithm that builds the chain code
//and derivative chain code for an object.Compute and display(command line or output text
//	file) both codes(chain code and derivative chain code) for an image with a single object.
void buildChainCode() {
	Mat img = imread("Images/triangle_up.bmp", IMREAD_COLOR);

	int arrDirI[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int arrDirJ[] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	std::vector<int> chaincode;
	std::vector<int> derivative;

	int dir = 7;
	std::pair<int, int> p1, p2, current, prev;

	bool found = false;
	for (int i = 0; i < img.rows && !found; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j) != Vec3b(255, 255, 255)) {
				p1 = { i, j };
				p2 = p1;
				found = true;
				break;
			}
		}
	}

	if (!found) return;

	current = p1;
	prev = p1;

	int ok = true;
	int n = 0;

	while (!(current == p2 && prev == p1 && n >= 2)) {
		n++;
		dir = dir % 2 == 0 ? (dir + 7) % 8 : (dir + 6) % 8;
		while (!(isInside(current.first + arrDirI[dir], current.second + arrDirJ[dir], img.rows, img.cols) &&
			img.at<Vec3b>(current.first + arrDirI[dir], current.second + arrDirJ[dir]) != Vec3b(255, 255, 255))) {
			dir = (dir + 1) % 8;
		}

		prev = current;
		current = { current.first + arrDirI[dir], current.second + arrDirJ[dir] };

		if (ok) {
			p2 = current;
			ok = false;
		}

		chaincode.push_back(dir);
	}

	for (int i = 0; i < chaincode.size(); i++) {
		if (i == 0) {
			derivative.push_back((chaincode[i] - 7 + 8) % 8);
		}
		else {
			derivative.push_back((chaincode[i] - chaincode[i - 1] + 8) % 8);
		}
	}

	FILE* f = fopen("result.txt", "w");

	fprintf(f, "Chain code: ");
	for (int i = 0; i < chaincode.size(); i++) {
		fprintf(f, "%d ", chaincode[i]);
	}

	fprintf(f, "\nDerivative: ");
	for (int i = 0; i < derivative.size(); i++) {
		fprintf(f, "%d ", derivative[i]);
	}

	fclose(f);
}

//Implement a function that reconstructs(draws) the border of an object over an image having
//as inputs the start point coordinates and the chain code in 8 - neighborhood(reconstruct.txt).
//Load the image gray_background.bmp and apply the function that reconstructs the border.
//You should obtain the contour of the word “EXCELLENT”(having all the letters
//connected).

void borderReconstruction() {
	Mat img = imread("Images/gray_background.bmp", IMREAD_COLOR);

	FILE* f = fopen("Images/reconstruct.txt", "r");

	int arrDirI[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int arrDirJ[] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	int x, y;

	fscanf(f, "%d", &x);
	fscanf(f, "%d", &y);

	int n;
	fscanf(f, "%d", &n);

	for (int i = 0; i < n; i++) {
		Point center(y, x);
		Scalar color(0, 0, 255);
		int radius = 1;
		int thickness = -1;

		circle(img, center, radius, color, thickness);

		int dir;
		fscanf(f, "%d", &dir);

		x += arrDirI[dir];
		y += arrDirJ[dir];
	}

	fclose(f);

	imshow("img", img);
	waitKey(0);
}

int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");


		//lab 1
		printf(" 13 - Change the gray level of image using additive factor\n");
		printf(" 14 - Change the gray level of image using multiplicative factor\n");
		printf(" 15 - Create a color image of dimension 256 x 256. Divide it into 4 squares and color the squares from top to bottom, left to right as : white, red, green, yellow.\n");
		printf(" 16 - Create a 3x3 float matrix, determine its inverse and print it.\n");

		//Lab 2
		printf(" 17 - Create a function that will copy the R, G and B channels of a color, RGB24 image(CV_8UC3 type) into three matrices of type CV_8UC1(grayscale images).Display these matrices in three distinct windows.\n");
		printf(" 18 - Create a function that will convert a color RGB24 image (CV_8UC3 type) to a grayscale image (CV_8UC1), and display the result image in a destination window.\n");
		printf(" 19 - Create a function for converting from grayscale to black and white (binary), using (2.2). Read the threshold from the console. Test the operation on multiple images, and using multiple thresholds.\n");
		printf(" 20 - Create a function that will compute the H, S and V values from the R, G, B channels of an image, using the equations from 2.6. Store each value (H, S, V) in a CV_8UC1 matrix. Display these matrices in distinct windows. Check the correctness of your implementation using the example below.\n");

		//Lab 3
		printf(" 21 - Compute the histogram for a given grayscale image (in an array of integers having dimension 256).\n");
		printf(" 22 - Compute the PDF (in an array of floats of dimension 256).\n");
		printf(" 23 - Compute the histogram for a given number of bins m ≤ 256.\n");
		printf(" 24 - Implement the multilevel thresholding algorithm from section.\n");
		printf(" 25 - Enhance the multilevel thresholding algorithm using the Floyd-Steinberg dithering from section 3.4.\n");
		printf(" 26 - Perform multilevel thresholding on a color image by applying the procedure from 3.3 on the Hue channel from the HSV color-space representation of the image. Modify only the Hue values, keeping the S and V channels unchanged or setting them to their maximum possible value. Transform the result back to RGB color-space for viewing.\n");

		//Lab 4
		printf(" 27 - For a specific object in a labeled image selected by a mouse click, compute the object’s area, center of mass, axis of elongation, perimeter, thinness ratio and aspect ratio.\n");
		printf(" 28 - Create a new processing function which takes as input a labeled image and keeps in the output image only the objects that have specific area and orientation.\n");

		//Lab 5
		printf(" 29 - Implement the breadth first traversal component labeling algorithm(Algorithm 1).You should be able to easily switch between the neighborhood types of 4 and 8.\n");
		printf(" 30 - Implement the two - pass component labeling algorithm.\n");

		//Lab 6
		printf(" 31 - Implement the border tracing algorithm and draw the object contour on an image having a single object.\n");
		printf(" 32 - Implement a function which generates a color image from a label matrix by assigning a random color to each label.Display the results.\n");
		printf(" 33 - Implement a function that reconstructs(draws) the border of an object over an image having as inputs the start point coordinates and the chain code in 8 - neighborhood.\n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				testBGR2HSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;

			//lab 1
			case 13:
				changeGrayLevelUsingAdditiveFactor();
				break;
			case 14:
				changeGrayLevelUsingMultiplicativeFactor();
				break;
			case 15:
				createImageWithFourColors();
				break;
			case 16:
				computeMatrixInverse();
				break;

			//Lab 2
			case 17:
				imageTo3GrayscaleImages();
				break;
			case 18:
				colorImageToGrayscaleImage();
				break;
			case 19:
				grayscaleToBlackAndWhite();
				break;
			case 20:
				displayHSV();
				break;

			//Lab 3
			case 21:
				displayHistogram(computeHistogram, 256);
				break;
			case 22:
				displayPDF();
				break;
			case 23:
				displayHistogram(computeHistogram, 100);
				break;
			case 24:
				displayImgAfterMultilevelThresholdingAlgorithm();
				break;
			case 25:
				displayImgAfterMultilevelThresholdingAlgorithmWithFloydSteinberg();
				break;
			case 26:
				displayMultilevelThresholdingOnColorImage();
				break;

			//Lab 4
			case 27:
				mouseClickObjectProperties();
				break;
			case 28:
				displayFilteredObjects();
				break;

			//Lab5
			case 29:
				generateColorImage();
				break;
			case 30:
				generateColorImageTwoPasses();
				break;

			//Lab6
			case 31:
				displayBorderTracingAlgorithm();
				break;
			case 32:
				buildChainCode();
				break;
			case 33:
				borderReconstruction();
				break;

		}
	}
	while (op!=0);
	return 0;
}