#include <iostream>

using namespace std;

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <fstream>

//constexpr auto C1 = (float) (0.01 * 255 * 0.01 * 255);
//constexpr auto C2 = (float) (0.03 * 255 * 0.03  * 255);

using namespace std;
using namespace cv;


/*
* reference: https://en.wikipedia.org/wiki/Daubechies_wavelet
*            https://en.wikipedia.org/wiki/Fast_wavelet_transform
*/
/* db filter data from: http://wavelets.pybytes.com/wavelet/db3/#coeffs */
static double db2_Lo_D[4] = { -0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025 };
static double db2_Hi_D[4] = { -0.48296291314469025, 0.836516303737469, -0.22414386804185735, -0.12940952255092145 };
static double db2_Lo_R[4] = { 0.48296291314469025, 0.836516303737469, 0.22414386804185735, -0.12940952255092145 };
static double db2_Hi_R[4] = { -0.12940952255092145, -0.22414386804185735, 0.836516303737469, -0.48296291314469025 };

static double db3_Lo_D[6] = { 0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569 };
static double db3_Hi_D[6] = { -0.3326705529509569, 0.8068915093133388, -0.4598775021193313, -0.13501102001039084, 0.08544127388224149, 0.035226291882100656 };
static double db3_Lo_R[6] = { 0.3326705529509569, 0.8068915093133388, 0.4598775021193313, -0.13501102001039084, -0.08544127388224149, 0.035226291882100656 };
static double db3_Hi_R[6] = { 0.035226291882100656, 0.08544127388224149, -0.13501102001039084, -0.4598775021193313, 0.8068915093133388, -0.3326705529509569 };

namespace qm
{
#define C1 (float) (0.01 * 255 * 0.01  * 255)
#define C2 (float) (0.03 * 255 * 0.03  * 255)


    // sigma on block_size
    double sigma(Mat& m, int i, int j, int block_size)
    {
        double sd = 0;

        Mat m_tmp = m(Range(i, i + block_size), Range(j, j + block_size));
        Mat m_squared(block_size, block_size, CV_64F);

        multiply(m_tmp, m_tmp, m_squared);

        // E(x)
        double avg = mean(m_tmp)[0];
        // E(x²)
        double avg_2 = mean(m_squared)[0];


        sd = sqrt(avg_2 - avg * avg);

        return sd;
    }

    // Covariance
    double cov(Mat& m1, Mat& m2, int i, int j, int block_size)
    {
        Mat m3 = Mat::zeros(block_size, block_size, m1.depth());
        Mat m1_tmp = m1(Range(i, i + block_size), Range(j, j + block_size));
        Mat m2_tmp = m2(Range(i, i + block_size), Range(j, j + block_size));


        multiply(m1_tmp, m2_tmp, m3);

        double avg_ro = mean(m3)[0]; // E(XY)
        double avg_r = mean(m1_tmp)[0]; // E(X)
        double avg_o = mean(m2_tmp)[0]; // E(Y)


        double sd_ro = avg_ro - avg_o * avg_r; // E(XY) - E(X)E(Y)

        return sd_ro;
    }

    // Mean squared error
    double eqm(Mat& img1, Mat& img2)
    {
        int i, j;
        double eqm = 0;
        int height = img1.rows;
        int width = img1.cols;

        for (i = 0; i < height; i++)
            for (j = 0; j < width; j++)
                eqm += (img1.at<double>(i, j) - img2.at<double>(i, j)) * (img1.at<double>(i, j) - img2.at<double>(i, j));

        eqm /= height * width;

        return eqm;
    }



    /**
     *	Compute the PSNR between 2 images
     */
    double psnr(Mat& img_src, Mat& img_compressed)
    {
        int D = 255;
        return (10 * log10((D * D) / eqm(img_src, img_compressed)));
    }


    /**
     * Compute the SSIM between 2 images
     */
    double ssim(Mat& img_src, Mat& img_compressed, int block_size, bool show_progress = false)
    {
        double ssim = 0;

        int nbBlockPerHeight = img_src.rows / block_size;
        int nbBlockPerWidth = img_src.cols / block_size;

        for (int k = 0; k < nbBlockPerHeight; k++)
        {
            for (int l = 0; l < nbBlockPerWidth; l++)
            {
                int m = k * block_size;
                int n = l * block_size;

                double avg_o = mean(img_src(Range(k, k + block_size), Range(l, l + block_size)))[0];
                double avg_r = mean(img_compressed(Range(k, k + block_size), Range(l, l + block_size)))[0];
                double sigma_o = sigma(img_src, m, n, block_size);
                double sigma_r = sigma(img_compressed, m, n, block_size);
                double sigma_ro = cov(img_src, img_compressed, m, n, block_size);

                ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));

            }
            // Progress
            if (show_progress)
                cout << "\r>>SSIM [" << (int)((((double)k) / nbBlockPerHeight) * 100) << "%]";
        }
        ssim /= nbBlockPerHeight * nbBlockPerWidth;

        if (show_progress)
        {
            cout << "\r>>SSIM [100%]" << endl;
            cout << "SSIM : " << ssim << endl;
        }

        return ssim;
    }

    /*
    void compute_quality_metrics(char* file1, char* file2, int block_size)
    {

        Mat img_src;
        Mat img_compressed;

        // Loading pictures
        img_src = imread(file1, CV_LOAD_IMAGE_GRAYSCALE);
        img_compressed = imread(file2, CV_LOAD_IMAGE_GRAYSCALE);


        img_src.convertTo(img_src, CV_64F);
        img_compressed.convertTo(img_compressed, CV_64F);

        int height_o = img_src.rows;
        int height_r = img_compressed.rows;
        int width_o = img_src.cols;
        int width_r = img_compressed.cols;

        // Check pictures size
        if (height_o != height_r || width_o != width_r)
        {
            cout << "Images must have the same dimensions" << endl;
            return;
        }

        // Check if the block size is a multiple of height / width
        if (height_o % block_size != 0 || width_o % block_size != 0)
        {
            cout << "WARNING : Image WIDTH and HEIGHT should be divisible by BLOCK_SIZE for the maximum accuracy" << endl
                << "HEIGHT : " << height_o << endl
                << "WIDTH : " << width_o << endl
                << "BLOCK_SIZE : " << block_size << endl
                << endl;
        }

        double ssim_val = ssim(img_src, img_compressed, block_size);
        double psnr_val = psnr(img_src, img_compressed, block_size);

        cout << "SSIM : " << ssim_val << endl;
        cout << "PSNR : " << psnr_val << endl;
    }*/
    
}

struct WT2Ceof
{
    /* layout of ceofs, common for DWT2 and SWT2
    *
    *     |  LF  |  HH  |
    *     ---------------
    *     |  HV  |  HD  |
    *
    */
    Mat LF;     /* low frequency */
    Mat HH;     /* high frequency in horizontal */
    Mat HV;     /* high frequency in vertical   */
    Mat HD;     /* high frequency in diagonal   */
};

enum class SAMPLE_AXIS
{
    SAMPLE_AXIS_X = 1,
    SAMPLE_AXIS_Y = 2
};

/* this is a helper function */
static void Upsample(const Mat& src, Mat& dst, SAMPLE_AXIS axis, bool needPadding, int offset)
{
    /* pad a row or col of zeros at the last if there was a row or col lost during the DWT downsample */
    int padding = needPadding ? 1 : 0;

    /* offset could only be 0 or 1, means insert 0 to the even or odd position */
    if (offset != 0 && offset != 1)
    {
        cerr << "Error: offset could only be 0 or 1" << endl;
        exit(1);
    }

    if (axis == SAMPLE_AXIS::SAMPLE_AXIS_X)
    {
        Mat img = Mat::zeros(src.rows, src.cols * 2 + padding, src.type());

        for (int j = 0; j < src.cols; j++)
        {
            src.col(j).copyTo(img.col(j * 2 + offset));
        }

        dst = img.clone();
    }
    else if (axis == SAMPLE_AXIS::SAMPLE_AXIS_Y)
    {
        Mat img = Mat::zeros(src.rows * 2 + padding, src.cols, src.type());

        for (int i = 0; i < src.rows; i++)
        {
            src.row(i).copyTo(img.row(i * 2 + offset));
        }

        dst = img.clone();
    }
    else
    {
        cerr << "Error: invalid axis!" << endl;
        exit(1);
    }

    return;
}

/* keep the even elements, dstlen = (srclen + filterLen - 1) / 2 */
static void DownSample(const Mat& src, Mat& dst, SAMPLE_AXIS axis)
{
    Mat img;

    if (axis == SAMPLE_AXIS::SAMPLE_AXIS_X)
    {
        img = Mat::zeros(src.rows, src.cols / 2, src.type());

        for (int i = 0; i < img.cols; i++)
        {
            src.col(i * 2 + 1).copyTo(img.col(i));
        }
    }
    else if (axis == SAMPLE_AXIS::SAMPLE_AXIS_Y)
    {
        img = Mat::zeros(src.rows / 2, src.cols, src.type());

        for (int i = 0; i < img.rows; i++)
        {
            src.row(i * 2 + 1).copyTo(img.row(i));
        }
    }
    else
    {
        cerr << "Error: invalid SAMPLE_AXIS" << endl;
        exit(1);
    }

    dst = img;
}

/* 2D DWT */
vector<WT2Ceof>  DWT2(const Mat& srcSignal, int dbn, int level)
{
    vector<WT2Ceof> ceofs;
    /* db filter decomposition kernel */
    Mat kernel_db_Lo_D, kernel_db_Hi_D;

    if (dbn == 3) { kernel_db_Lo_D = Mat(1, 6, CV_64F, db3_Lo_D); kernel_db_Hi_D = Mat(1, 6, CV_64F, db3_Hi_D); }
    else if (dbn == 2) { kernel_db_Lo_D = Mat(1, 4, CV_64F, db2_Lo_D); kernel_db_Hi_D = Mat(1, 4, CV_64F, db2_Hi_D); }
    else { cout << "Error : only support db2 and db3" << endl; exit(1); }

    assert(kernel_db_Lo_D.cols == kernel_db_Lo_D.cols);

    Mat LF = srcSignal; // the low frequncy ceof that need to be decomposited

    for (int i = 0; i < level; i++)
    {
        int ceofWidth = (LF.cols + kernel_db_Lo_D.cols - 1) / 2;
        int ceofHeight = (LF.rows + kernel_db_Lo_D.cols - 1) / 2;

        WT2Ceof ceof;
        /* step 1: apply filter to each row*/
        Mat srcSignal_ext, mat_Lo_D, mat_Hi_D;

        /* border extropolation, add kernel.cols - 1 to the left */
        // top bottom left right
        copyMakeBorder(LF, srcSignal_ext, 0, 0, kernel_db_Lo_D.cols - 1, 0, BORDER_REFLECT);

        filter2D(srcSignal_ext, mat_Lo_D, CV_64F, kernel_db_Lo_D, Point(0, 0), 0.0, BORDER_REFLECT);
        filter2D(srcSignal_ext, mat_Hi_D, CV_64F, kernel_db_Hi_D, Point(0, 0), 0.0, BORDER_REFLECT);

        /* downsample at x-axis */
        DownSample(mat_Lo_D, mat_Lo_D, SAMPLE_AXIS::SAMPLE_AXIS_X);
        DownSample(mat_Hi_D, mat_Hi_D, SAMPLE_AXIS::SAMPLE_AXIS_X);

        assert(mat_Lo_D.cols == ceofWidth);

        /* step 2: apply filter to each col */
        Mat packedRowFiltedMat(mat_Lo_D.rows, ceofWidth * 2, mat_Lo_D.type());

        mat_Lo_D.colRange(0, mat_Lo_D.cols).copyTo(packedRowFiltedMat.colRange(0, ceofWidth));
        mat_Hi_D.colRange(0, mat_Hi_D.cols).copyTo(packedRowFiltedMat.colRange(ceofWidth, packedRowFiltedMat.cols));

        copyMakeBorder(packedRowFiltedMat, packedRowFiltedMat, kernel_db_Lo_D.cols - 1, 0, 0, 0, BORDER_REFLECT);

        Mat mat_packed_Lo_D, mat_packed_Hi_D;

        filter2D(packedRowFiltedMat, mat_packed_Lo_D, CV_64F, kernel_db_Lo_D.t(), Point(0, 0), 0.0, BORDER_REFLECT);
        filter2D(packedRowFiltedMat, mat_packed_Hi_D, CV_64F, kernel_db_Hi_D.t(), Point(0, 0), 0.0, BORDER_REFLECT);

        /* downsample at y-axis  */
        DownSample(mat_packed_Lo_D, mat_packed_Lo_D, SAMPLE_AXIS::SAMPLE_AXIS_Y);
        DownSample(mat_packed_Hi_D, mat_packed_Hi_D, SAMPLE_AXIS::SAMPLE_AXIS_Y);

        assert(mat_packed_Lo_D.rows == ceofHeight);

        /* now we have all the ceofs */
        ceof.LF = (mat_packed_Lo_D.rowRange(0, mat_packed_Lo_D.rows).colRange(0, mat_packed_Lo_D.cols / 2)).clone();
        ceof.HH = (mat_packed_Lo_D.rowRange(0, mat_packed_Lo_D.rows).colRange(mat_packed_Lo_D.cols / 2, mat_packed_Lo_D.cols)).clone();
        ceof.HV = (mat_packed_Hi_D.rowRange(0, mat_packed_Hi_D.rows).colRange(0, mat_packed_Hi_D.cols / 2)).clone();
        ceof.HD = (mat_packed_Hi_D.rowRange(0, mat_packed_Hi_D.rows).colRange(mat_packed_Hi_D.cols / 2, mat_packed_Hi_D.cols)).clone();

        ceofs.push_back(ceof);
        LF = ceof.LF;
    }

    return ceofs;
}

/*
* return image's format is double (CV_64F), need to convert to srcImg's type like this
*
*   recImg.convertTo(displayImg, srcType);
*/
Mat IDWT2(vector<WT2Ceof> ceofs, int dbn, int level, int dstWidth, int dstHeight)
{
    if (ceofs.size() != level)
    {
        cerr << "Error: ceofs.size() != level " << endl;
        exit(1);
    }

    /* db filter reconstruction kernel */
    Mat kernel_db_Lo_R, kernel_db_Hi_R;

    if (dbn == 3) { kernel_db_Lo_R = Mat(1, 6, CV_64F, db3_Lo_R); kernel_db_Hi_R = Mat(1, 6, CV_64F, db3_Hi_R); }
    else if (dbn == 2) { kernel_db_Lo_R = Mat(1, 4, CV_64F, db2_Lo_R); kernel_db_Hi_R = Mat(1, 4, CV_64F, db2_Hi_R); }
    else { cout << "Error : only support db2 and db3" << endl; exit(1); }

    assert(kernel_db_Lo_R.cols == kernel_db_Hi_R.cols);

    Mat recImg;
    int recImgWidth, recImgHeight;

    for (int i = level - 1; i >= 0; i--)
    {
        if (i > 0)
        {
            recImgWidth = ceofs.at(i - 1).LF.cols;
            recImgHeight = ceofs.at(i - 1).LF.rows;
        }
        else
        {
            recImgWidth = dstWidth;
            recImgHeight = dstHeight;
        }

        /* step1 upsample rows */
        WT2Ceof ceofsUpsampledMat;

        // when (recImgHeight + kernel_db_Lo_R.cols - 1) is odd, the last last row is lost, need to add it back
        bool needPadding = ceofs.at(i).LF.rows * 2 - (kernel_db_Lo_R.cols - 1) < recImgHeight;

        Upsample(ceofs.at(i).LF, ceofsUpsampledMat.LF, SAMPLE_AXIS::SAMPLE_AXIS_Y, needPadding, 1);
        Upsample(ceofs.at(i).HH, ceofsUpsampledMat.HH, SAMPLE_AXIS::SAMPLE_AXIS_Y, needPadding, 1);
        Upsample(ceofs.at(i).HV, ceofsUpsampledMat.HV, SAMPLE_AXIS::SAMPLE_AXIS_Y, needPadding, 1);
        Upsample(ceofs.at(i).HD, ceofsUpsampledMat.HD, SAMPLE_AXIS::SAMPLE_AXIS_Y, needPadding, 1);
        //Upsample(Mat::zeros(ceofs.at(i).HH.rows, ceofs.at(i).HH.cols, ceofs.at(i).HH.type()), ceofsUpsampledMat.HH, SAMPLE_AXIS::SAMPLE_AXIS_Y, needPadding, 1);
        //Upsample(Mat::zeros(ceofs.at(i).HV.rows, ceofs.at(i).HV.cols, ceofs.at(i).HV.type()), ceofsUpsampledMat.HV, SAMPLE_AXIS::SAMPLE_AXIS_Y, needPadding, 1);
        //Upsample(Mat::zeros(ceofs.at(i).HD.rows, ceofs.at(i).HD.cols, ceofs.at(i).HD.type()), ceofsUpsampledMat.HD, SAMPLE_AXIS::SAMPLE_AXIS_Y, needPadding, 1);

        /* step2 convole at y- axis (cols) */
        Mat LF, HH, HV, HD;
        filter2D(ceofsUpsampledMat.LF, LF, CV_64F, kernel_db_Lo_R.t(), Point(0, 0), 0.0, BORDER_REFLECT);
        filter2D(ceofsUpsampledMat.HH, HH, CV_64F, kernel_db_Lo_R.t(), Point(0, 0), 0.0, BORDER_REFLECT);
        filter2D(ceofsUpsampledMat.HV, HV, CV_64F, kernel_db_Hi_R.t(), Point(0, 0), 0.0, BORDER_REFLECT);
        filter2D(ceofsUpsampledMat.HD, HD, CV_64F, kernel_db_Hi_R.t(), Point(0, 0), 0.0, BORDER_REFLECT);

        LF = LF + HV;
        HH = HH + HD;

        /* here pick from 0 becasue the anchor point is (0, 0), the first element is valid,
           the last filterLen - 1 elements are caculated from extended values
        */
        assert(LF.rows = recImgHeight + kernel_db_Lo_R.cols - 1);

        LF = LF.rowRange(0, recImgHeight);
        HH = HH.rowRange(0, recImgHeight);

        /* step3 upsample cols */
        needPadding = LF.cols * 2 - (kernel_db_Lo_R.cols - 1) < recImgWidth;

        Upsample(LF, LF, SAMPLE_AXIS::SAMPLE_AXIS_X, needPadding, 1);
        Upsample(HH, HH, SAMPLE_AXIS::SAMPLE_AXIS_X, needPadding, 1);

        /* step4 convole at x- axis (rows) */
        Mat loMat, hiMat;
        filter2D(LF, loMat, CV_64F, kernel_db_Lo_R, Point(0, 0), 0.0, BORDER_REFLECT);
        filter2D(HH, hiMat, CV_64F, kernel_db_Hi_R, Point(0, 0), 0.0, BORDER_REFLECT);

        assert(loMat.cols = recImgWidth + kernel_db_Lo_R.cols - 1);

        loMat = loMat.colRange(0, recImgWidth);
        hiMat = hiMat.colRange(0, recImgWidth);

        /* last step: merge low and high */
        recImg = loMat + hiMat;

        if (i > 0)
        {
            /* in a multipass reconstruction, we should use the reconstructed image as the LF of next pass,
               but not the decomposited one
            */
            ceofs.at(i - 1).LF = recImg;
        }
    }

    return recImg;
}

void ShowCeof(WT2Ceof& ceofs, string prefix, int type)
{
    Mat LF, HH, HV, HD;

    ceofs.LF.convertTo(LF, type);
    ceofs.HH.convertTo(HH, type);
    ceofs.HV.convertTo(HV, type);
    ceofs.HD.convertTo(HD, type);

    imshow(prefix + "LF", LF);
    imshow(prefix + "HH", HH);
    imshow(prefix + "HV", HV);
    imshow(prefix + "HD", HD);
}

Mat SubRangle(Mat input, int top, int bottom, int left, int right)
{
    assert(top >= 0 && top < bottom&& bottom <= input.rows);
    assert(left >= 0 && left < right&& right <= input.cols);

    Mat output = Mat(bottom - top, right - left, input.type());

    for (int i = 0; i < output.rows; i++)
    {
        input.row(i + top).colRange(left, right).copyTo(output.row(i));
    }

    return output;
}

/*  Stationary discrete 2-D wavelet transform
*
*   how to calculate the coefficients
*
*         row          col
*   src -------*Lo_D -------*Lo_D----->LF (Approximation)
*
*         row          col
*   src -------*Lo_D -------*Hi_D----->HH (horizontal)
*
*         row          col
*   src -------*Hi_D -------*Lo_D----->HV (vertical)
*
 *         row          col
*   src -------*Hi_D -------*Hi_D----->HD (diagonal )
*/

vector<WT2Ceof> SWT2(const Mat& srcSignal, int dbn, int level)
{
    vector<WT2Ceof> ceofs;

    /* db filter decomposition kernel */
    Mat kernel_db_Lo_D, kernel_db_Hi_D;

    if (dbn == 3) { kernel_db_Lo_D = Mat(1, 6, CV_64F, db3_Lo_D); kernel_db_Hi_D = Mat(1, 6, CV_64F, db3_Hi_D); }
    else if (dbn == 2) { kernel_db_Lo_D = Mat(1, 4, CV_64F, db2_Lo_D); kernel_db_Hi_D = Mat(1, 4, CV_64F, db2_Hi_D); }
    else { cout << "Error : only support db2 and db3" << endl; exit(1); }

    assert(kernel_db_Lo_D.cols == kernel_db_Hi_D.cols);

    int minEdge = min(srcSignal.cols, srcSignal.rows);
    int filerLen = kernel_db_Lo_D.cols;

    for (int i = 0; i < level; i++)
    {
        if (filerLen > minEdge)
        {
            cout << "Error : sginal length should > 2 * filter length" << endl;
            assert(0);
            exit(-1);
        }

        filerLen *= 2;
    }

    Mat matInput = srcSignal;

    for (int i = 0; i < level; i++)
    {
        Mat input_ext;
        Mat lpRowMat, hpRowMat;
        WT2Ceof ceof;

        Point anchor = Point(-1, -1);

        /* step 1: apply filter to each row*/
        /* Opencv filter2D function doesn't support wrap extend, so need to do it mannually */
        // int wrapPadding = kernel_db_Lo_D.cols - paddingMargin;
        int wrapPadding = kernel_db_Lo_D.cols / 2;

        copyMakeBorder(matInput, input_ext, 0, 0, wrapPadding, wrapPadding, BORDER_WRAP);

        filter2D(input_ext, lpRowMat, CV_64F, kernel_db_Lo_D, anchor, 0.0, BORDER_ISOLATED);
        filter2D(input_ext, hpRowMat, CV_64F, kernel_db_Hi_D, anchor, 0.0, BORDER_ISOLATED);

        /* step 2: apply filter to each col */
        copyMakeBorder(lpRowMat, lpRowMat, wrapPadding, wrapPadding, 0, 0, BORDER_WRAP);
        copyMakeBorder(hpRowMat, hpRowMat, wrapPadding, wrapPadding, 0, 0, BORDER_WRAP);

        filter2D(lpRowMat, ceof.LF, CV_64F, kernel_db_Lo_D.t(), anchor, 0.0, BORDER_ISOLATED);
        filter2D(lpRowMat, ceof.HH, CV_64F, kernel_db_Hi_D.t(), anchor, 0.0, BORDER_ISOLATED);
        filter2D(hpRowMat, ceof.HV, CV_64F, kernel_db_Lo_D.t(), anchor, 0.0, BORDER_ISOLATED);
        filter2D(hpRowMat, ceof.HD, CV_64F, kernel_db_Hi_D.t(), anchor, 0.0, BORDER_ISOLATED);

        ceof.LF = SubRangle(ceof.LF, wrapPadding, lpRowMat.rows - wrapPadding, wrapPadding, lpRowMat.cols - wrapPadding) * 0.5;
        ceof.HH = SubRangle(ceof.HH, wrapPadding, lpRowMat.rows - wrapPadding, wrapPadding, lpRowMat.cols - wrapPadding) * 0.5;
        ceof.HV = SubRangle(ceof.HV, wrapPadding, lpRowMat.rows - wrapPadding, wrapPadding, lpRowMat.cols - wrapPadding) * 0.5;
        ceof.HD = SubRangle(ceof.HD, wrapPadding, lpRowMat.rows - wrapPadding, wrapPadding, lpRowMat.cols - wrapPadding) * 0.5;

        assert(matInput.rows == ceof.LF.rows && matInput.cols == ceof.LF.cols);

        ceofs.push_back(ceof);

        /* step 3: upsample the filters */
        if (i < level - 1)
        {
            Upsample(kernel_db_Lo_D, kernel_db_Lo_D, SAMPLE_AXIS::SAMPLE_AXIS_X, 0, 0);
            Upsample(kernel_db_Hi_D, kernel_db_Hi_D, SAMPLE_AXIS::SAMPLE_AXIS_X, 0, 0);

            matInput = ceof.LF;
        }
    }

    return ceofs;
}

Mat ISWT2(vector<WT2Ceof>& ceofs, int dbn, int level)
{
    Mat dstImg;

    /* db filter reconstruction kernel */
    Mat kernel_db_Lo_R, kernel_db_Hi_R;

    if (dbn == 3) { kernel_db_Lo_R = Mat(1, 6, CV_64F, db3_Lo_R); kernel_db_Hi_R = Mat(1, 6, CV_64F, db3_Hi_R); }
    else if (dbn == 2) { kernel_db_Lo_R = Mat(1, 4, CV_64F, db2_Lo_R); kernel_db_Hi_R = Mat(1, 4, CV_64F, db2_Hi_R); }
    else { cout << "Error : only support db2 and db3" << endl; exit(1); }

    /* build the filters */
    vector<Mat> loR_kernels, hiR_kernels;

    loR_kernels.push_back(kernel_db_Lo_R);
    hiR_kernels.push_back(kernel_db_Hi_R);

    int paddingMargin = 1;

    for (int i = 1; i < level; i++)
    {
        Upsample(kernel_db_Lo_R, kernel_db_Lo_R, SAMPLE_AXIS::SAMPLE_AXIS_X, 0, 0);
        Upsample(kernel_db_Hi_R, kernel_db_Hi_R, SAMPLE_AXIS::SAMPLE_AXIS_X, 0, 0);

        loR_kernels.push_back(kernel_db_Lo_R);
        hiR_kernels.push_back(kernel_db_Hi_R);

        paddingMargin *= 2;
    }

    Mat extLF, extHH, extHV, extHD;
    Mat tempLF, tempHH, tempHV, tempHD, LoMat, HiMat;

    dstImg = ceofs.at(level - 1).LF.clone();

    for (int i = level - 1; i >= 0; i--)
    {
        kernel_db_Lo_R = loR_kernels.at(i);
        kernel_db_Hi_R = hiR_kernels.at(i);

        // int wrapPadding = kernel_db_Lo_R.cols - paddingMargin;
        int wrapPadding = kernel_db_Lo_R.cols / 2;

        Point anchor = Point(-1, -1);

        copyMakeBorder(dstImg, extLF, wrapPadding, wrapPadding, 0, 0, BORDER_WRAP);
        copyMakeBorder(ceofs.at(i).HH, extHH, wrapPadding, wrapPadding, 0, 0, BORDER_WRAP);
        copyMakeBorder(ceofs.at(i).HV, extHV, wrapPadding, wrapPadding, 0, 0, BORDER_WRAP);
        copyMakeBorder(ceofs.at(i).HD, extHD, wrapPadding, wrapPadding, 0, 0, BORDER_WRAP);

        filter2D(extLF, tempLF, CV_64F, kernel_db_Lo_R.t(), anchor, 0.0, BORDER_ISOLATED);
        filter2D(extHH, tempHH, CV_64F, kernel_db_Hi_R.t(), anchor, 0.0, BORDER_ISOLATED);
        filter2D(extHV, tempHV, CV_64F, kernel_db_Lo_R.t(), anchor, 0.0, BORDER_ISOLATED);
        filter2D(extHD, tempHD, CV_64F, kernel_db_Hi_R.t(), anchor, 0.0, BORDER_ISOLATED);

        Mat extLo, extHi;

        copyMakeBorder(tempLF + tempHH, extLo, 0, 0, wrapPadding, wrapPadding, BORDER_WRAP);
        copyMakeBorder(tempHV + tempHD, extHi, 0, 0, wrapPadding, wrapPadding, BORDER_WRAP);

        filter2D(extLo, LoMat, CV_64F, kernel_db_Lo_R, anchor, 0.0, BORDER_ISOLATED);
        filter2D(extHi, HiMat, CV_64F, kernel_db_Hi_R, anchor, 0.0, BORDER_ISOLATED);

        dstImg = SubRangle(LoMat + HiMat, wrapPadding + paddingMargin, LoMat.rows - wrapPadding + paddingMargin,
            wrapPadding + paddingMargin, LoMat.cols - wrapPadding + paddingMargin) * 0.5;

        paddingMargin /= 2;
    }

    return dstImg;
}

vector<WT2Ceof> denoise_gray(vector<WT2Ceof> coefs, int dbn, int level, int width, int height, double thereshold) {
    vector<WT2Ceof> ret;
    WT2Ceof new_coef;
    for (int l = 0; l < level; l++) {
        Mat coef_LF = coefs.at(l).LF.clone();
        Mat coef_HH = coefs.at(l).HH.clone();
        Mat coef_HV = coefs.at(l).HV.clone();
        Mat coef_HD = coefs.at(l).HD.clone();

        int coef_w = coef_HH.cols;
        int coef_h = coef_HH.rows;
        int type = coef_HH.type();

        int HH_cnt = 0;
        int HV_cnt = 0;
        for (int i = 0; i < coef_h; i++) {
            for (int j = 0; j < coef_w; j++) {
                if (abs(coef_HH.at<double>(i, j)) < thereshold) {
                    coef_HH.at<double>(i, j) = 0;
                    HH_cnt++;
                }
                if (abs(coef_HV.at<double>(i, j)) < thereshold) {
                    coef_HV.at<double>(i, j) = 0;
                    HV_cnt++;
                }
                if (abs(coef_HD.at<double>(i, j)) < thereshold) {
                    coef_HD.at<double>(i, j) = 0;
                }
            }
        }

        new_coef.LF = coef_LF.clone();
        new_coef.HH = coef_HH.clone();
        //new_coef.HH = Mat::zeros(coef_h, coef_w, type);
        new_coef.HV = coef_HV.clone();
        //new_coef.HV = Mat::zeros(coef_h, coef_w, type);
        new_coef.HD = coef_HD.clone();
        //new_coef.HD = Mat::zeros(coef_h, coef_w, type);

        ret.push_back(new_coef);

    }
    return ret;
}

vector<WT2Ceof> denoise_color(vector<WT2Ceof> coefs, int dbn, int level, int width, int height, double thereshold) {
    vector<WT2Ceof> ret;
    WT2Ceof new_coef;
    for (int l = 0; l < level; l++) {
        Mat coef_LF = coefs.at(l).LF.clone();
        Mat coef_HH = coefs.at(l).HH.clone();
        Mat coef_HV = coefs.at(l).HV.clone();
        Mat coef_HD = coefs.at(l).HD.clone();

        int coef_w = coef_HH.cols;
        int coef_h = coef_HH.rows;
        int type = coef_HH.type();

        int HH_cnt = 0;
        int HV_cnt = 0;
        for (int i = 0; i < coef_h; i++) {
            for (int j = 0; j < coef_w; j++) {
                for (int c = 0; c < 3; c++) {
                    if (abs(coef_HH.at<Vec3d>(i, j)[c]) < thereshold) {
                        coef_HH.at<Vec3d>(i, j)[c] = 0;
                        HH_cnt++;
                    }
                    if (abs(coef_HV.at<Vec3d>(i, j)[c]) < thereshold) {
                        coef_HV.at<Vec3d>(i, j)[c] = 0;
                        HV_cnt++;
                    }
                    if (abs(coef_HD.at<Vec3d>(i, j)[c]) < thereshold) {
                        coef_HD.at<Vec3d>(i, j)[c] = 0;
                    }
                }
                
            }
        }

        new_coef.LF = coef_LF.clone();
        new_coef.HH = coef_HH.clone();
        //new_coef.HH = Mat::zeros(coef_h, coef_w, type);
        new_coef.HV = coef_HV.clone();
        //new_coef.HV = Mat::zeros(coef_h, coef_w, type);
        new_coef.HD = coef_HD.clone();
        //new_coef.HD = Mat::zeros(coef_h, coef_w, type);

        /*if (l != 0) {
            new_coef.HH = Mat::zeros(coef_h, coef_w, type);
            new_coef.HV = Mat::zeros(coef_h, coef_w, type);
            new_coef.HD = Mat::zeros(coef_h, coef_w, type);
        }*/

        ret.push_back(new_coef);

    }
    return ret;
}


vector<WT2Ceof> denoise_adapt_gray(vector<WT2Ceof> coefs, int dbn, int level, int width, int height) {
    vector<WT2Ceof> ret;
    WT2Ceof new_coef;
    for (int l = 0; l < level; l++) {
        Mat coef_LF = coefs.at(l).LF.clone();
        Mat coef_HH = coefs.at(l).HH.clone();
        Mat coef_HV = coefs.at(l).HV.clone();
        Mat coef_HD = coefs.at(l).HD.clone();

        int coef_w = coef_HH.cols;
        int coef_h = coef_HH.rows;
        int type = coef_HH.type();

        int HH_cnt = 0;
        int HV_cnt = 0;
        
        //
        double thereshold = 0.5;

        for (int i = 0; i < coef_h; i++) {
            for (int j = 0; j < coef_w; j++) {
                if (abs(coef_HH.at<double>(i, j)) < thereshold) {
                    coef_HH.at<double>(i, j) = 0;
                    HH_cnt++;
                }
                if (abs(coef_HV.at<double>(i, j)) < thereshold) {
                    coef_HV.at<double>(i, j) = 0;
                    HV_cnt++;
                }
                if (abs(coef_HD.at<double>(i, j)) < thereshold) {
                    coef_HD.at<double>(i, j) = 0;
                }
            }
        }

        //cout << coef_HH << endl;
        //cout << coef_HH << endl;
        // have no change???? w/o .clone coef is the ref of coefs[x]
        /*if (countNonZero(coef_HH != coefs.at(l).HH) == 0) {
            cout << "hwrong" << endl;
        }
        if (countNonZero(coef_HV != coefs.at(l).HV) == 0) {
            cout << "vwrong" << endl;
        }
        if (countNonZero(coef_HD != coefs.at(l).HD) == 0) {
            cout << "dwrong" << endl;
        }*/

        new_coef.LF = coef_LF.clone();
        new_coef.HH = coef_HH.clone();
        //new_coef.HH = Mat::zeros(coef_h, coef_w, type);
        new_coef.HV = coef_HV.clone();
        //new_coef.HV = Mat::zeros(coef_h, coef_w, type);
        new_coef.HD = coef_HD.clone();
        //new_coef.HD = Mat::zeros(coef_h, coef_w, type);

        /*if (l != 0) {
            new_coef.HH = Mat::zeros(coef_h, coef_w, type);
            new_coef.HV = Mat::zeros(coef_h, coef_w, type);
            new_coef.HD = Mat::zeros(coef_h, coef_w, type);
        }*/

        ret.push_back(new_coef);

    }
    return ret;
}



double universal_thresh(double sigma, double width, double height) {
    // log_e()
    return sigma * sqrt(2 * log(width * height));
}


double soft(double data, double thereshold) {
    // thereshold must > 0
    // if magnitude(data) < 0 thereshold, set to 0
    // otherwise data = data * (1 - abs(thereshold / data))
    if (abs(data) <= thereshold) {
        return 0;
    }
    return data > 0 ? (data - thereshold) : (data + thereshold);
}

double hard(double data, double thereshold) {
    // thereshold must > 0
    // if magnitude(data) < 0 thereshold, set to 0
    // otherwise data
    if (abs(data) <= thereshold) {
        return 0;
    }
    return data;
}

void soft_thereshold_perlevel(WT2Ceof& coefs, int dbn, int width, int height, double sigma) {
    // perform thereshold on each image channel of specific level **inplace**
    double var = pow(sigma, 2);
    // VisuShrink
    double thereshold = universal_thresh(sigma, width, height);
    
    int cols = coefs.HH.cols;
    int rows = coefs.HH.rows;
    
    // for color image
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int c = 0; c < 3; c++) {
                coefs.HH.at<Vec3d>(i, j)[c] = soft(coefs.HH.at<Vec3d>(i, j)[c], thereshold);
                coefs.HV.at<Vec3d>(i, j)[c] = soft(coefs.HV.at<Vec3d>(i, j)[c], thereshold);
                coefs.HD.at<Vec3d>(i, j)[c] = soft(coefs.HD.at<Vec3d>(i, j)[c], thereshold);
            }
        }
    }
}

vector<WT2Ceof> VisuShrink(vector<WT2Ceof> coefs, int dbn, int level, int width, int height, double sigma) {
    // C++ version of VisuShrink
    vector<WT2Ceof> ret;

    for (int l = 0; l < level; l++) {
        WT2Ceof new_coef;
        new_coef.LF = coefs.at(l).LF.clone();
        new_coef.HH = coefs.at(l).HH.clone();
        new_coef.HV = coefs.at(l).HV.clone();
        new_coef.HD = coefs.at(l).HD.clone();
        soft_thereshold_perlevel(new_coef, dbn, width, height, sigma);
        ret.push_back(new_coef);
    }

    return ret;
}

vector<WT2Ceof> Bayes(vector<WT2Ceof> coefs, int dbn, int level, int width, int height, vector<double> sigmas) {
    // C++ version of VisuShrink
    vector<WT2Ceof> ret;

    for (int l = 0; l < level; l++) {
        WT2Ceof new_coef;
        new_coef.LF = coefs.at(l).LF.clone();
        new_coef.HH = coefs.at(l).HH.clone();
        new_coef.HV = coefs.at(l).HV.clone();
        new_coef.HD = coefs.at(l).HD.clone();
        soft_thereshold_perlevel(new_coef, dbn, width, height, sigmas[l]);
        ret.push_back(new_coef);
    }

    return ret;
}

vector<double> getAdatpSigma_HD(vector<WT2Ceof> coefs, int level) {
    // cal sigma as the median of the nonzero abs(value) in each coef level
    // only consider the HD coef
    vector<double> sigmas(level);
    for (int l = 0; l < level; l++) {
        vector<double> non_zero_coefs;
        int cols = coefs.at(l).HD.cols;
        int rows = coefs.at(l).HD.rows;
        cout << "level " << l << " cols " << cols << " rows " << rows << endl;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int c = 0; c < 3; c++) {
                    double value = coefs.at(l).HD.at<Vec3d>(i, j)[c];
                    if (value != 0) {
                        non_zero_coefs.push_back(abs(value));
                    }
                }
            }
        }
        sort(non_zero_coefs.begin(), non_zero_coefs.end());
        int len = non_zero_coefs.size();
        if (len % 2 == 0) {
            sigmas[l] = (non_zero_coefs[len / 2] + non_zero_coefs[len / 2 - 1]) / 2;
        }
        else {
            sigmas[l] = non_zero_coefs[len / 2];
        }
        cout << "level " << l << " sigma " << sigmas.back() << endl;
    }
    return sigmas;
}

vector<double> getAdatpSigma_H(vector<WT2Ceof> coefs, int level) {
    // cal sigma as the median of the nonzero abs(value) in each coef level
    // consider all high frequency coef
    vector<double> sigmas(level);
    for (int l = 0; l < level; l++) {
        vector<double> non_zero_coefs;
        int cols = coefs.at(l).HD.cols;
        int rows = coefs.at(l).HD.rows;
        cout << "level " << l << " cols " << cols << " rows " << rows << endl;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int c = 0; c < 3; c++) {
                    double value_HH = coefs.at(l).HH.at<Vec3d>(i, j)[c];
                    double value_HV = coefs.at(l).HV.at<Vec3d>(i, j)[c];
                    double value_HD = coefs.at(l).HD.at<Vec3d>(i, j)[c];
                    if (value_HH != 0) {
                        non_zero_coefs.push_back(abs(value_HH));
                    }
                    if (value_HV != 0) {
                        non_zero_coefs.push_back(abs(value_HV));
                    }
                    if (value_HD != 0) {
                        non_zero_coefs.push_back(abs(value_HD));
                    }
                }
            }
        }
        sort(non_zero_coefs.begin(), non_zero_coefs.end());
        int len = non_zero_coefs.size();
        if (len % 2 == 0) {
            sigmas[l] = (non_zero_coefs[len / 2] + non_zero_coefs[len / 2 - 1]) / 2;
        }
        else {
            sigmas[l] = non_zero_coefs[len / 2];
        }
        cout << "level " << l << " sigma " << sigmas[l] << endl;
    }
    return sigmas;
}

double psnr(Mat& original, Mat& processed) {
    int cols = original.cols;
    int rows = original.rows;
    Mat tmp(rows, cols, CV_64F);
    original.convertTo(original, CV_64F);
    processed.convertTo(processed, CV_64F);
    cv::subtract(original, processed, tmp);
    multiply(tmp, tmp, tmp);
    return 10 * log10(255 * 255 / cv::mean(tmp).val[0]);
}

Mat ReadRawImageAs64FC3(const string& filename, int width, int height) {
    // raw texture img is saved as float32 x channel 3 RGB, range 0~1
    // read and convert to float64 x channel 3 RGB, range 0~1
    float* pRawBuff = new float[(size_t)height * width * 3 * sizeof(float)];
    FILE* fp = nullptr;
    fopen_s(&fp, filename.c_str(), "rb");
    if (fp) {
        errno_t err_code = fread(pRawBuff, 1, ((size_t)height * width * 3 * sizeof(float)), fp);
        fclose(fp);
        fp = nullptr;
    }
    else {
        cout << "File open failed!" << endl;
    }
    Mat image(height, width, CV_32FC3, pRawBuff);
    normalize(image, image, 0, 1, NORM_MINMAX);
    Mat image_ret(height, width, CV_64FC3);
    image.convertTo(image_ret, CV_64FC3);
    return image_ret;
}

Mat Convert64FC3To8UC3(const Mat& src, int width, int height) {
    // change data range from 0~1 To 0~255 channel 3 BGR
    Mat tmp = src * 255;
    Mat image_ret(height, width, CV_8UC3);
    tmp.convertTo(image_ret, CV_8UC3);
    cvtColor(image_ret, image_ret, COLOR_RGB2BGR);
    return image_ret;
}

int main() {
    string filename("./_raw/Train0/GN.raw");
    int width = 1920;
    int height = 1080;
    Mat srcImg = ReadRawImageAs64FC3(filename, width, height);
    Mat image_disp = Convert64FC3To8UC3(srcImg, width, height);
    imshow("original", image_disp);

    //string filename("colorflower_noisy.png");
    //Mat srcImg = imread(filename, IMREAD_COLOR);

    int srcType = srcImg.type();
    //int width = srcImg.cols;// x
    //int height = srcImg.rows;// y
    int level = 3;
    int dbn = 3;

    vector<WT2Ceof> coefs = DWT2(srcImg, dbn, level);

    vector<WT2Ceof> denoise_coefs;

    vector<double> sigmas = getAdatpSigma_HD(coefs, level);
    cout << "sigma " << sigmas[0] << endl;
    double sigma = sigmas[0] / 2;
    // denoise_coef = denoise_color(ceofs, dbn, level, width, height, thereshold);
    denoise_coefs = VisuShrink(coefs, dbn, level, width, height, sigma);
    // denoise_coefs = coefs;

    Mat denoiseImg = IDWT2(denoise_coefs, dbn, level, width, height);

    Mat den;

    Mat image_denoise_disp = Convert64FC3To8UC3(denoiseImg, width, height);
    imshow("denoised", image_denoise_disp);

    //denoiseImg.convertTo(den, srcType);

    //imshow("denoiseImg", den);

    double psnr_v = psnr(srcImg, denoiseImg);
    cout << "psnr" << psnr_v << endl;
    
    //cout << sizeof(float) << endl;
    //system("pause");
    
    waitKey();
    destroyAllWindows();

    imwrite("denoised_image.png", image_denoise_disp);


	return 0;
}