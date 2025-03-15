// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>

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
				newImg.at<unsigned char>(i, j + 1) = newImg.at<unsigned char>(i, j + 1) + (int)(7 * error / 16.0);
			}
			if (i + 1 < newImg.rows && j - 1 > 0) {
				newImg.at<unsigned char>(i + 1, j - 1) = newImg.at<unsigned char>(i + 1, j - 1) + (int)(3 * error / 16.0);
			}
			if (i + 1 < newImg.rows) {
				newImg.at<unsigned char>(i + 1, j) = newImg.at<unsigned char>(i + 1, j) + (int)(5 * error / 16.0);
			}
			if (i + 1 < newImg.rows && j + 1 < newImg.cols) {
				newImg.at<unsigned char>(i + 1, j + 1) = newImg.at<unsigned char>(i + 1, j + 1) + (int)(error / 16.0);
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

	imgH = multilevelThresholdingAlgorithm(imgH);

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
				pixel = Vec3b((0 + m) * 255, (X + m) * 255, (C + m) * 255);
			}
			else if (H >= 60 && H < 120) {
				pixel = Vec3b((0 + m) * 255, (C + m) * 255, (X + m) * 255);
			}
			else if (H >= 120 && H < 180) {
				pixel = Vec3b((X + m) * 255, (C + m) * 255, (0 + m) * 255);
			}
			else if (H >= 180 && H < 240) {
				pixel = Vec3b((C + m) * 255, (X + m) * 255, (0 + m) * 255);
			}
			else if (H >= 240 && H < 300) {
				pixel = Vec3b((C + m) * 255, (0 + m) * 255, (X + m) * 255);
			}
			else if (H >= 300 && H < 360) {
				pixel = Vec3b((X + m) * 255, (0 + m) * 255, (C + m) * 255);
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
		}
	}
	while (op!=0);
	return 0;
}