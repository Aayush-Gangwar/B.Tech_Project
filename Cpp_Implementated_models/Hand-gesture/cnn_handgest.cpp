// #include <cmath>
// #include "hand_gesture_weight.h"
// using namespace std;
#include "cnn.h"

double relu(double x)
{
    return (x < 0) ? 0 : x;
}

void softmax(const double input[], int size, double output[]) {
     double max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
        sum_exp += std::exp(input[i] - max_val);
    }

    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val) / sum_exp;
        }

}


void convolution(const double flattenedImage[], const int kernels[], double output[], int imageWidth, int imageHeight, int imageDepth, int k_h, int k_w, const int biases[], int numKernels , char a_f)
{
    int output_h = imageHeight - k_h + 1;
    int output_w = imageWidth - k_w + 1;
    // int output_d = k_d;

    for (int k = 0; k < numKernels; ++k)
    {
        for (int i = 0; i < output_h; ++i)
        {
            for (int j = 0; j < output_w; ++j)
            {

                output[k * output_h * output_w + i * output_w + j] = biases[k]; // Initialize output with bias for the current kernel

                for (int kd = 0; kd < imageDepth; ++kd)
                {
                    for (int ki = 0; ki < k_h; ++ki)
                    {
                        for (int kj = 0; kj < k_w; ++kj)
                        {
                            output[k * output_h * output_w + i * output_w + j] +=
                                // flattenedImage[(i + ki) * k_w * imageDepth + (j + kj)*imageDepth + kd] * kernels[k * (k_h * k_w) + ki * k_w + kj + kd];
                                flattenedImage[ (j+kj) + (i+ki)*imageWidth + kd*imageHeight*imageWidth]  * kernels[k * (k_h * k_w*imageDepth) + ki * k_w + kd*k_h*k_w+kj];
                        }
                    }
                }
                if(a_f=='R'){
                    output[k * output_h * output_w + i * output_w + j] =  relu(output[k * output_h * output_w + i * output_w + j]);
                }
            }
        }
    }
}

void maxPooling(const double image[], double output[], int imageWidth, int numChannels, int pool_size) {
    int i_h = (imageWidth);
    int i_w = i_h;
    int output_h = i_h / pool_size;
    int output_w = i_w / pool_size;

    for (int c = 0; c < numChannels; ++c) {
        for (int i = 0; i < output_h; ++i) {
            for (int j = 0; j < output_w; ++j) {
                double max_val = 0.0;
                for (int pi = 0; pi < pool_size;pi+=2) {
                    for (int pj = 0; pj < pool_size; pj+=2) {
                        max_val = fmax(max_val, image[c * (i_h * i_w) + (i * pool_size + pi) * i_w + (j * pool_size + pj)]);
                    }
                }
                output[c * (output_h * output_w) + i * output_w + j] = max_val;
            }
        }
    }
}


void fullyConnectedLayer(const double input[], double output[], const int weights[], const int bias[], int inputSize, int outputSize, char a_f) {
    for (int i = 0; i < outputSize; ++i) {
        output[i] = bias[i];
        for (int j = 0; j < inputSize; ++j) {
            output[i] += input[j] * weights[i * inputSize + j];
        }

        if (a_f=='R') {
            output[i] = (output[i] < 0) ? 0 : output[i];
        }
    }
}


void CNN(const double flattenedImage[], int imageWidth, int imageHeight, double output[]) {
    // Convolutional layer
    int num_kernels1 = 8;
    int kernelWidth1 = 5;
    int kernelHeight1 = 5;
    int imagedepth1 = 1;
    char a_f = 'R';
    double conv1Output[8*46*46];
    convolution(flattenedImage, convolution1_weights, conv1Output, imageWidth, imageHeight, imagedepth1, kernelHeight1, kernelWidth1 , convolution1_bias,num_kernels1, a_f);
   // Max pooling layer
    int pool_size = 2;
    double pool1Output[23*23*8];
    maxPooling(conv1Output, pool1Output, imageWidth - kernelWidth1 + 1 ,num_kernels1, pool_size);
    int num_kernels2 = 16;
    int kernelWidth2 = 3;
    int kernelHeight2 = 3;
    int imageWidth2=23;
    int imageHeight2 = 23;
    int imagedepth2 = 8;
    double conv2Output[21*21*16];
    convolution(pool1Output, convolution2_weights, conv2Output, imageWidth2, imageHeight2,imagedepth2, kernelHeight2, kernelWidth2 , convolution2_bias,num_kernels2,a_f);
    // Max pooling layer 2
    double pool2Output[10*10*16];
    maxPooling(conv2Output, pool2Output, imageWidth2 - kernelWidth2 + 1,num_kernels2, pool_size);
    // merge fully connected layer. 
    char a_f2='R';
    double fully1output[256];
    fullyConnectedLayer(pool2Output,fully1output,dense1_weights,dense1_bias,1600,256,a_f2);
    char a_f3='S';
    double fully2output[10];
    fullyConnectedLayer(fully1output,fully2output,dense2_weights,dense2_bias,256,10,a_f3);
    if(a_f3=='S'){
        softmax(fully2output,10,output);
    }
    // for (int i = 0; i < 10; ++i) {
    //     output[i] = fully2output[i];
    // }
}




