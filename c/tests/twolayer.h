#include "../ebnn.h"
uint8_t l_conv_pool_bn_bst0_bconv_W[4] = {17,127,94,127};
float l_conv_pool_bn_bst0_bconv_b[2] = {-0.0058961543,0.005744196};
float l_conv_pool_bn_bst0_bn_gamma[2] = {1.0009999,0.82154155};
float l_conv_pool_bn_bst0_bn_beta[2] = {0.20623535,0.00065535656};
float l_conv_pool_bn_bst0_bn_mean[2] = {-0.6608156,-0.014598005};
float l_conv_pool_bn_bst0_bn_std[2] = {1.2920483,0.50866258};
void l_conv_pool_bn_bst0(float* input, uint8_t* output){
  fconv_layer(input, l_conv_pool_bn_bst0_bconv_W, output, l_conv_pool_bn_bst0_bconv_b, l_conv_pool_bn_bst0_bn_gamma, l_conv_pool_bn_bst0_bn_beta, l_conv_pool_bn_bst0_bn_mean, l_conv_pool_bn_bst0_bn_std, 1, 2, 28, 28, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
}

uint8_t l_b_conv_pool_bn_bst1_bconv_W[8] = {93,255,73,255,3,255,105,127};
float l_b_conv_pool_bn_bst1_bconv_b[2] = {0.010726526,0.020384386};
float l_b_conv_pool_bn_bst1_bn_gamma[2] = {0.96458465,1.0010142};
float l_b_conv_pool_bn_bst1_bn_beta[2] = {0.13672742,-0.12987822};
float l_b_conv_pool_bn_bst1_bn_mean[2] = {3.7213533,1.0021703};
float l_b_conv_pool_bn_bst1_bn_std[2] = {2.2130439,4.5835638};
void l_b_conv_pool_bn_bst1(uint8_t* input, uint8_t* output){
  bconv_layer(input, l_b_conv_pool_bn_bst1_bconv_W, output, l_b_conv_pool_bn_bst1_bconv_b, l_b_conv_pool_bn_bst1_bn_gamma, l_b_conv_pool_bn_bst1_bn_beta, l_b_conv_pool_bn_bst1_bn_mean, l_b_conv_pool_bn_bst1_bn_std, 1, 2, 28, 28, 2, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
}

float l_b_linear_bn_softmax2_bl_b[10] = {0.0021179421,0.00081903388,0.00053207856,-0.00076088007,0.0030896282,0.00055891689,-0.00078302022,-0.00089003262,0.00093711104,0.0010822912};
uint8_t l_b_linear_bn_softmax2_bl_W[1960] = {89,211,208,26,133,47,173,81,223,140,130,239,157,8,101,213,239,13,14,217,156,214,64,191,21,135,130,58,230,192,4,172,24,129,124,249,23,224,109,233,246,70,209,111,240,28,195,254,4,96,63,235,230,131,253,6,224,127,95,183,75,224,159,225,24,3,26,161,193,79,67,134,255,167,164,82,96,189,224,112,249,224,165,193,224,157,161,74,147,247,151,145,253,217,245,159,51,231,246,38,0,22,74,80,0,20,152,52,130,42,80,8,146,45,91,76,129,27,12,144,151,228,171,36,204,82,144,5,178,194,200,126,224,32,61,245,194,118,31,253,24,249,63,226,120,67,221,41,192,31,246,250,1,254,13,160,13,161,254,0,127,81,96,47,198,238,82,226,27,240,13,35,107,162,13,12,222,202,83,122,180,253,36,251,132,82,255,253,103,64,42,144,140,0,7,12,226,72,52,46,3,127,98,254,191,223,239,85,255,71,90,227,77,207,149,188,40,133,102,134,66,128,128,157,220,56,101,150,37,139,18,60,113,226,47,227,12,27,216,240,224,50,239,46,4,104,113,226,127,206,28,153,118,17,193,196,199,50,30,250,99,65,14,197,78,51,219,191,163,213,135,176,74,172,92,72,127,35,130,213,119,53,9,90,223,50,202,75,63,238,163,184,147,254,203,95,114,39,194,228,0,208,250,246,204,152,66,178,140,190,214,127,67,255,212,96,9,199,18,32,12,76,128,96,136,7,0,4,40,70,3,33,8,64,99,40,153,12,17,66,192,214,70,26,8,65,150,80,216,112,173,13,226,165,131,24,77,32,22,0,5,67,112,116,64,54,10,10,136,225,65,98,28,0,230,9,80,135,46,50,225,97,223,246,41,128,3,196,120,0,3,251,79,255,254,3,251,255,216,132,162,189,97,0,6,186,138,9,26,243,244,22,68,160,196,169,80,195,189,60,131,231,52,32,63,207,192,78,254,210,0,251,244,35,15,255,147,149,243,242,229,211,5,159,236,66,44,233,160,64,223,116,0,9,78,66,0,121,188,36,14,125,82,25,181,191,131,216,79,106,190,91,247,255,114,142,189,232,254,119,112,2,249,84,136,4,184,16,48,0,64,0,1,19,40,53,64,160,42,3,128,126,160,4,79,191,130,112,183,229,12,175,186,200,141,213,208,204,35,255,242,131,159,255,180,190,197,255,143,24,15,80,232,1,246,28,0,9,144,166,1,192,56,128,120,7,100,95,72,241,251,240,15,203,119,41,255,187,239,255,191,135,243,243,254,255,255,110,176,95,243,254,3,255,223,160,255,99,162,65,222,0,76,32,88,1,176,3,41,176,12,153,44,134,68,7,207,4,19,189,93,89,238,35,239,5,0,31,41,113,150,176,14,79,148,50,255,184,0,14,127,128,33,255,248,24,79,255,5,159,253,192,63,47,92,0,200,67,244,29,0,39,249,120,127,255,16,0,255,242,72,31,255,10,196,159,240,128,231,84,176,238,235,200,220,248,154,93,15,104,32,44,255,194,115,198,221,216,74,57,199,224,31,249,160,135,167,55,207,236,44,85,189,96,206,156,25,62,62,50,255,76,94,199,247,88,0,247,223,66,143,255,242,80,223,250,32,15,252,180,22,254,7,218,31,128,127,147,248,31,241,30,7,218,107,176,255,192,125,95,248,132,243,119,207,79,64,175,164,254,8,253,15,224,15,208,254,64,116,15,244,31,160,255,160,243,13,253,220,0,254,253,168,15,243,209,211,255,244,32,186,124,0,6,151,255,174,175,191,44,181,39,252,191,230,255,239,255,85,255,191,120,43,249,47,221,255,195,248,47,254,23,19,127,156,190,85,249,23,72,142,14,59,208,195,228,250,0,152,91,40,225,81,96,6,0,71,98,4,209,253,32,31,247,236,134,254,191,247,247,178,175,158,183,31,223,64,21,14,248,69,192,111,192,61,1,248,5,210,31,196,180,5,255,124,122,191,127,231,175,127,251,236,0,40,2,0,4,8,50,4,16,53,132,94,211,115,141,187,246,249,111,183,110,233,255,217,107,3,247,119,192,61,236,233,131,234,210,112,126,70,39,143,130,1,240,240,138,159,15,167,85,249,201,166,247,141,255,63,238,95,111,243,171,239,62,191,61,228,232,186,42,48,7,192,16,130,40,5,64,1,3,56,0,38,143,192,227,100,48,0,61,49,0,5,32,56,0,0,1,80,233,184,107,240,1,137,132,42,79,243,25,99,27,46,187,2,242,125,4,77,95,90,136,249,76,96,7,24,7,186,48,18,127,227,170,15,255,20,112,63,255,247,3,255,190,240,125,121,142,199,215,141,255,252,0,223,255,232,9,254,213,134,19,246,32,90,159,174,77,13,117,38,193,180,209,127,151,178,169,82,222,69,124,69,96,90,237,60,79,86,96,215,254,12,43,178,7,146,152,24,0,231,179,0,50,144,81,102,255,128,52,140,144,133,57,244,10,131,193,232,55,120,209,135,255,17,128,63,240,151,36,255,38,188,3,240,159,128,31,137,120,0,119,205,240,0,135,151,0,5,24,248,9,163,143,129,145,48,91,35,21,0,244,195,201,58,110,127,64,210,35,235,15,206,31,255,199,228,255,16,126,39,236,15,26,238,30,220,66,249,254,78,126,64,73,40,0,77,179,0,11,30,232,64,38,166,171,96,29,84,148,8,31,164,138,120,76,163,89,214,69,47,224,56,81,238,129,84,255,246,40,94,239,247,170,186,93,80,129,13,20,34,65,6,210,72,104,16,106,78,134,141,49,181,75,16,52,93,31,133,166,200,49,97,54,207,128,31,200,92,3,219,33,249,255,72,47,255,200,33,81,110,44,161,32,153,139,0,5,71,118,106,16,63,127,255,243,255,255,255,29,127,255,245,255,223,255,221,219,255,164,6,127,189,235,1,245,13,144,40,13,199,66,0,25,192,192,26,99,40,0,49,91,192,31,24,122,2,122,239,112,122,177,246,141,146,45,121,223,128,199,255,193,62,251,250,1,239,125,64,223,143,234,19,212,124,0,191,255,224,3,255,245,72,175,252,177,152,14,14,77,160,0,112,47,240,36,199,183,40,0,203,42,92,232,238,204,160,146,0,0,147,0,2,5,107,240,2,45,239,252,254,127,255,125,243,240,187,244,50,96,36,149,104,8,160,7,0,41,69,201,225,202,62,127,74,32,95,240,223,173,190,62,15,167,193,223,180,114,15,77,84,53,20,128,74,218,148,120,94,158,203,34,102,111,108,214,203,248,22,35,93,45,252,213,213,53,107,234,64,8,249,166,64,0,67,160,2,3,131,1,95,77,134,30,247,18,197,253,190,51,205,5,202,71,134,2,16,82,50,223,192,33,238,255,1,101,187,252,124,142,255,233,220,223,245,191,175,238,248,136,242,11,137,79,176,62,176,252,0,180,223,64,5,87,163,0,219,212,192,23,245,88,0,123,129,112,73,195,146,49,196,13,20,130,19,40,157,4,88,11,97,105,160,58,208,68,124,237,2,255,254,192,3,255,254,69,40,0,25,72,1,145,0,240,23,251,218,10,255,253,223,224,127,137,169,2,103,44,84,113,179,93,124,90,32,119,152,117,128,248,14,4,252,129,208,188,43,21,2,195,23,255,252,7,255,159,192,126,228,232,15,235,161,3,250,230,64,63,21,4,7,132,48,0,255,3,8,47,184,52,30,174,157,18,240,21,145,176,3,166,25,32,66,228,253,255,169,3,127,238,201,234,77,191,255,126,255,208,252,185,62,127,191,100,214,171,161,119,23,64,11,215,1,30,70,2,175,245,252,83,252,215,182,223,246,227,23,156,191,73,253,59,254,255,255,255,123,255,127,243,255,255,223,184,127,255,216,7,190,69,224,51,131,37,3,220,20,72,255,149,100,63,254,216,37,248,214,160,127,15,150,25,253,224,2,63,62,224,8,103,211,32,195,254,224,128,58,255,108,0,1,245,213,63,207,10,102,151,6,184,16,0,19,144,16,2,128,47,234,14,130,6,146,193,128,125,12,236,3,83,147,64,59,159,96,215,236,225,127,17,140,1,140,5,193,225,4,0,42,4,130,5,192,28,195,116,51,166,199,178,157,90,247,85,211,49,180,160,102,125,68,76,171,162,251,72,2,253,252,52,85,183,205,209,127,177,247,47,132,231,160,0,71,49,123,86,246,114,249,251,94,16,0,25,252,48,3,94,112,155,253,88,160,12,223,244,135,6,188,0,146,113,38,1,129,128,254,0,32,31,224,32,7,254,44,0,55,232,130,191,109,224,35,233,61,192,255,131,250,95,231,220,7,255,207,185,215,255,112,5,127,194,4,15,218,98,2,104,234,79,5,239,194,69,192,18,187,240,6,197,112,193,5,250,208,34,89,230,84,123,255,120,113,191,239,255,255,254};
float l_b_linear_bn_softmax2_bn_gamma[10] = {1.0010118,1.0010545,1.0009764,1.000946,1.0010223,1.001035,1.0009689,1.001002,1.0009592,1.0010301};
float l_b_linear_bn_softmax2_bn_beta[10] = {-0.034081127,0.019167693,0.01857039,0.049917269,-0.014301067,-0.039307147,-0.05770912,0.03103438,-0.00076180219,0.012194083};
float l_b_linear_bn_softmax2_bn_mean[10] = {76.321053,18.250515,40.425751,-34.31781,54.566586,13.393117,-20.267597,-30.648325,37.658245,40.055397};
float l_b_linear_bn_softmax2_bn_std[10] = {72.164726,71.086914,72.790321,68.775742,67.462059,56.947086,69.349663,65.214142,60.630417,59.372597};
void l_b_linear_bn_softmax2(uint8_t* input, uint8_t* output){
  blinear_layer(input, l_b_linear_bn_softmax2_bl_W, output, l_b_linear_bn_softmax2_bl_b, l_b_linear_bn_softmax2_bn_gamma, l_b_linear_bn_softmax2_bn_beta, l_b_linear_bn_softmax2_bn_mean, l_b_linear_bn_softmax2_bn_std, 1, 1568, 10);
}


uint8_t temp1[208] = {0};
uint8_t temp2[208] = {0};
void ebnn_compute(float *input, uint8_t *output){
  l_conv_pool_bn_bst0(input, temp1);
  l_b_conv_pool_bn_bst1(temp1, temp2);
  l_b_linear_bn_softmax2(temp2, output);
}
