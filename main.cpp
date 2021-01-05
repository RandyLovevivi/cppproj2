#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include "data.h"
#define SIZE 128
using namespace std;
using namespace cv;
int main() {
    Mat src ;
    src = imread("/Users/harrisonhu/Documents/computer science/program/c++/homework/proj2/SimpleCNNbyCPP-main/samples/face.jpg");
    if (src.empty()) {
        cout << "could not load image..." << endl;
        return -1;
    }
    else cout << "load succsessful." << endl;
    imshow("input", src);
    clock_t time_start=clock();
    float picture[SIZE+2][SIZE+2][3]={0};

    for (int row = 0; row < SIZE; row++) {
        for (int col = 0; col < SIZE; col++) {
            int b = src.at<Vec3b>(row, col)[0];
            int g = src.at<Vec3b>(row, col)[1];
            int r = src.at<Vec3b>(row, col)[2];
            picture[row+1][col+1][0]=(float)r/255;
            picture[row+1][col+1][1]=(float)g/255;
            picture[row+1][col+1][2]=(float)b/255;
        }
    }

    //conv1 channels=3 padding=1 stride=2//
    float picture1[SIZE/2][SIZE/2][16]={0};
    int out_channels=16;
    int in_channels=3;

    for (int o = 0; o < out_channels; ++o) {
        for (int i = 0; i < in_channels; ++i) {
            int col1=0,row1=0;
            int colresult=0,rowresult=0;

            float kernel_oi_00 = conv0_weight[o*(in_channels*3*3) + i*(3*3) + 0];
            float kernel_oi_01 = conv0_weight[o*(in_channels*3*3) + i*(3*3) + 1];
            float kernel_oi_02 = conv0_weight[o*(in_channels*3*3) + i*(3*3) + 2];

            float kernel_oi_10 = conv0_weight[o*(in_channels*3*3) + i*(3*3) + 3];
            float kernel_oi_11 = conv0_weight[o*(in_channels*3*3) + i*(3*3) + 4];
            float kernel_oi_12 = conv0_weight[o*(in_channels*3*3) + i*(3*3) + 5];

            float kernel_oi_20 = conv0_weight[o*(in_channels*3*3) + i*(3*3) + 6];
            float kernel_oi_21 = conv0_weight[o*(in_channels*3*3) + i*(3*3) + 7];
            float kernel_oi_22 = conv0_weight[o*(in_channels*3*3) + i*(3*3) + 8];

            while (true) {

                float p1=picture[row1][col1][i];
                float p2=picture[row1][col1+1][i];
                float p3=picture[row1][col1+2][i];

                float p4=picture[row1+1][col1][i];
                float p5=picture[row1+1][col1+1][i];
                float p6=picture[row1+1][col1+2][i];

                float p7=picture[row1+2][col1][i];
                float p8=picture[row1+2][col1+1][i];
                float p9=picture[row1+2][col1+2][i];

                picture1[rowresult][colresult][o]+=kernel_oi_00*p1+kernel_oi_01*p2+kernel_oi_02*p3+kernel_oi_10*p4+kernel_oi_11*p5+kernel_oi_12*p6+kernel_oi_20*p7+kernel_oi_21*p8+kernel_oi_22*p9;

                col1+=2;
                colresult++;

                if(col1>127){
                    col1=0;
                    row1+=2;

                    rowresult++;
                    colresult=0;

                    if(row1>127){
                        break;
                    }
                }
            }
        }
        float bias_oi = conv0_bias[o];
        for (int x = 0; x < 64; x++) {
            for (int y = 0; y < 64; y++) {
                picture1[x][y][o]+=bias_oi;
            }
        }
    }

    //RELU
    for(int z=0;z<16;z++) {
        for (int x = 0; x < 64; x++) {
            for (int y = 0; y < 64; y++) {
                if(picture1[x][y][z]<0) picture1[x][y][z]=0;
            }
        }
    }

    //MAXPOOLING
    float picture2[32][32][16]={0};
    for(int k=0;k<16;k++) {
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                float a=picture1[i*2][j*2][k];
                float b=picture1[i*2][j*2+1][k];
                float c=picture1[i*2+1][j*2][k];
                float d=picture1[i*2+1][j*2+1][k];
                float temp=a;
                if(temp<b)temp=b;
                if(temp<c)temp=c;
                if(temp<d)temp=d;
                picture2[i][j][k]=temp;
            }
        }
    }


    //conv2 channels=3 padding=1 stride=1//
    float picture3[32+2][32+2][16]={0};
    for(int k=0;k<16;k++) {
        for (int i = 1; i < 33; i++) {
            for (int j = 1; j < 33; j++) {
                picture3[i][j][k]=picture2[i-1][j-1][k];
            }
        }
    }
    float picture4[32][32][32];
    out_channels=32;
    in_channels=16;
    for (int o = 0; o < out_channels; ++o) {
        for (int i = 0; i < in_channels; ++i) {
            int col1=0,row1=0;
            int colresult=0,rowresult=0;


            float kernel_oi_00 = conv1_weight[o*(in_channels*3*3) + i*(3*3) + 0];
            float kernel_oi_01 = conv1_weight[o*(in_channels*3*3) + i*(3*3) + 1];
            float kernel_oi_02 = conv1_weight[o*(in_channels*3*3) + i*(3*3) + 2];

            float kernel_oi_10 = conv1_weight[o*(in_channels*3*3) + i*(3*3) + 3];
            float kernel_oi_11 = conv1_weight[o*(in_channels*3*3) + i*(3*3) + 4];
            float kernel_oi_12 = conv1_weight[o*(in_channels*3*3) + i*(3*3) + 5];

            float kernel_oi_20 = conv1_weight[o*(in_channels*3*3) + i*(3*3) + 6];
            float kernel_oi_21 = conv1_weight[o*(in_channels*3*3) + i*(3*3) + 7];
            float kernel_oi_22 = conv1_weight[o*(in_channels*3*3) + i*(3*3) + 8];

            while (true) {

                float p1=picture3[row1][col1][i];
                float p2=picture3[row1][col1+1][i];
                float p3=picture3[row1][col1+2][i];

                float p4=picture3[row1+1][col1][i];
                float p5=picture3[row1+1][col1+1][i];
                float p6=picture3[row1+1][col1+2][i];

                float p7=picture3[row1+2][col1][i];
                float p8=picture3[row1+2][col1+1][i];
                float p9=picture3[row1+2][col1+2][i];

                picture4[rowresult][colresult][o]+=kernel_oi_00*p1+kernel_oi_01*p2+kernel_oi_02*p3+kernel_oi_10*p4+kernel_oi_11*p5+kernel_oi_12*p6+kernel_oi_20*p7+kernel_oi_21*p8+kernel_oi_22*p9;

                col1+=1;
                colresult++;

                if(col1>31){
                    col1=0;
                    row1+=1;

                    rowresult++;
                    colresult=0;

                    if(row1>31){
                        break;
                    }
                }
            }
        }
        float bias_oi = conv1_bias[o];
        for (int x = 0; x < 32; x++) {
            for (int y = 0; y < 32; y++) {
                picture4[x][y][o]+=bias_oi;
            }
        }
    }

    //RELU picture4
    for(int z=0;z<32;z++) {
        for (int x = 0; x < 32; x++) {
            for (int y = 0; y < 32; y++) {
                if(picture4[x][y][z]<0) picture4[x][y][z]=0;
            }
        }
    }

    //MAXPOLLING picture4
    float picture5[16][16][32]={0};
    for(int k=0;k<32;k++) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                float a=picture4[i*2][j*2][k];
                float b=picture4[i*2][j*2+1][k];
                float c=picture4[i*2+1][j*2][k];
                float d=picture4[i*2+1][j*2+1][k];
                float temp=a;
                if(temp<b)temp=b;
                if(temp<c)temp=c;
                if(temp<d)temp=d;
                picture5[i][j][k]=temp;
            }
        }
    }

    //conv3 channels=3 padding=1 stride=2//
    float picture6[16+2][16+2][32]={0};
    for(int k=0;k<16;k++) {
        for (int i = 1; i < 17; i++) {
            for (int j = 1; j < 17; j++) {
                picture6[i][j][k]=picture5[i-1][j-1][k];
            }
        }
    }

    float picture7[8][8][32];
    out_channels=32;
    in_channels=32;
    for (int o = 0; o < out_channels; ++o) {
        for (int i = 0; i < in_channels; ++i) {
            int col1=0,row1=0;
            int colresult=0,rowresult=0;

            float kernel_oi_00 = conv2_weight[o*(in_channels*3*3) + i*(3*3) + 0];
            float kernel_oi_01 = conv2_weight[o*(in_channels*3*3) + i*(3*3) + 1];
            float kernel_oi_02 = conv2_weight[o*(in_channels*3*3) + i*(3*3) + 2];

            float kernel_oi_10 = conv2_weight[o*(in_channels*3*3) + i*(3*3) + 3];
            float kernel_oi_11 = conv2_weight[o*(in_channels*3*3) + i*(3*3) + 4];
            float kernel_oi_12 = conv2_weight[o*(in_channels*3*3) + i*(3*3) + 5];

            float kernel_oi_20 = conv2_weight[o*(in_channels*3*3) + i*(3*3) + 6];
            float kernel_oi_21 = conv2_weight[o*(in_channels*3*3) + i*(3*3) + 7];
            float kernel_oi_22 = conv2_weight[o*(in_channels*3*3) + i*(3*3) + 8];

            while (true) {

                float p1=picture6[row1][col1][i];
                float p2=picture6[row1][col1+1][i];
                float p3=picture6[row1][col1+2][i];

                float p4=picture6[row1+1][col1][i];
                float p5=picture6[row1+1][col1+1][i];
                float p6=picture6[row1+1][col1+2][i];

                float p7=picture6[row1+2][col1][i];
                float p8=picture6[row1+2][col1+1][i];
                float p9=picture6[row1+2][col1+2][i];

                picture7[rowresult][colresult][o]+=kernel_oi_00*p1+kernel_oi_01*p2+kernel_oi_02*p3+kernel_oi_10*p4+kernel_oi_11*p5+kernel_oi_12*p6+kernel_oi_20*p7+kernel_oi_21*p8+kernel_oi_22*p9;

                col1+=2;
                colresult++;

                if(col1>15){
                    col1=0;
                    row1+=2;

                    rowresult++;
                    colresult=0;

                    if(row1>15){
                        break;
                    }
                }
            }
        }
        float bias_oi = conv2_bias[o];
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                picture7[x][y][o]+=bias_oi;
            }
        }
    }
    //RELU picture7
    for(int z=0;z<32;z++) {
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                if(picture7[x][y][z]<0) picture7[x][y][z]=0;
            }
        }
    }

    float output1=0,output2=0;
    for(int z=0;z<32;z++) {
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                output1+=picture7[x][y][z]*fc0_weight[y+x*8+z*8*8];
                output2+=picture7[x][y][z]*fc0_weight[2048+y+x*8+z*8*8];
            }
        }
    }

    float o1=exp(output1)/(exp(output1)+exp(output2));
    float o2=exp(output2)/(exp(output1)+exp(output2));

    cout<<"bg:"<<o1<<" "<<"face:"<<o2;
    cout<<endl;
    clock_t time_end=clock();
    cout<<"time use:"<<1000*(time_end-time_start)/(double)CLOCKS_PER_SEC<<"ms"<<endl;

    return 0;

}

