#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "test_input.h"
#include "cnn.h"
using namespace std;

int main() {
      int output_final[100];
    int ido=0;
    for(int k=0;k<100;k++){
    double flattenedImage[64*64*3];
    int idx = 0;
    for(int d=0;d<3;d++){
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            flattenedImage[idx++] = test_image[ido++];
        }
    }
    }
        double output[131];  // Assuming the output size is 10
        VGG16(flattenedImage, 64,64, output);
       auto maxElementIterator = std::max_element(output, output + 131);
        int maxIndex = std::distance(output, maxElementIterator);
        output_final[k]=maxIndex;
    }
    return 0;
}
