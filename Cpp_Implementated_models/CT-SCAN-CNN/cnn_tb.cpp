#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
// #include "ct_scan_cnn.cpp"
#include "test_image.h"
#include "cnn.h"
using namespace std;

int main() {
   
    // auto start = high_resolution_clock::now();
    //   ap_uint<64> begin_time = hls::tLastTimerValue()
    // #pragma HLS stable variable=begin_time
    int output_final[10];
    int ido=0;
    for(int k=0;k<10;k++){
    double flattenedImage[256*256];
    int idx = 0;
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 256; ++j) {
            flattenedImage[idx++] = test_image[ido++];
        }
    }
        double output[2];  // Assuming the output size is 10
        CNN(flattenedImage, 256,256, output);
       auto maxElementIterator = std::max_element(output, output + 2);
        int maxIndex = std::distance(output, maxElementIterator);
        output_final[k]=maxIndex;
    }

    //  ap_uint<64> end_time = hls::tLastTimerValue();
    // #pragma HLS stable variable=end_time

    // Calculate the execution time
    // ap_uint<64> execution_time = end_time - begin_time;

    // Print or use the execution time as needed
    // printf("Execution Time: %lld cycles\n", (long long)execution_time);
    // auto stop = high_resolution_clock::now();
    //  auto duration = duration_cast<microseconds>(stop - start);
    // cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    // int count =0;
    // for(int i=0;i<100;i++){
    //     // cout<<output_final[i]<<" "<<test_prediction[i]<<endl;
    //     if(output_final[i]!=test_prediction[i]){
    //         count++;
    //     }
    // }
    // cout<<count<<endl;
    return 0;
}
