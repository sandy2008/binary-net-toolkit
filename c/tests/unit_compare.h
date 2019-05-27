uint8_t binary_in[98] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,240,0,127,255,0,31,255,224,0,255,192,0,15,204,0,0,28,0,0,1,192,0,0,14,0,0,0,120,0,0,7,192,0,0,62,0,0,0,240,0,0,7,128,0,3,240,0,0,255,0,0,31,224,0,7,248,0,1,254,0,0,255,128,0,15,224,0,0,0,0,0,0,0,0,0,0,0,0};
uint8_t binary_in2[98] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,128,0,0,24,0,0,1,128,16,0,56,1,192,3,128,60,0,120,3,192,7,128,63,0,240,3,248,31,0,63,225,224,3,143,62,0,56,115,192,7,199,188,0,124,63,128,7,1,248,0,48,15,0,7,0,64,0,112,0,0,7,0,0,0,96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
float binary_out[784] = {28.0,28.0,28.0,28.0,28.0,14.0,-2.0,-4.0,8.0,12.0,22.0,22.0,22.0,20.0,18.0,18.0,20.0,20.0,16.0,12.0,12.0,12.0,12.0,10.0,14.0,28.0,28.0,28.0,28.0,28.0,28.0,28.0,28.0,14.0,-2.0,-4.0,8.0,12.0,22.0,22.0,22.0,20.0,18.0,18.0,20.0,20.0,16.0,12.0,12.0,12.0,12.0,10.0,14.0,28.0,28.0,28.0,28.0,28.0,28.0,28.0,28.0,14.0,-2.0,-4.0,8.0,12.0,22.0,22.0,22.0,20.0,18.0,18.0,20.0,20.0,16.0,12.0,12.0,12.0,12.0,10.0,14.0,28.0,28.0,28.0,28.0,28.0,28.0,28.0,28.0,14.0,-2.0,-4.0,8.0,12.0,22.0,22.0,22.0,20.0,18.0,18.0,20.0,20.0,16.0,12.0,12.0,12.0,12.0,10.0,14.0,28.0,28.0,28.0,28.0,28.0,28.0,28.0,28.0,14.0,-2.0,-4.0,8.0,12.0,22.0,22.0,22.0,20.0,18.0,18.0,20.0,20.0,16.0,12.0,12.0,12.0,12.0,10.0,14.0,28.0,28.0,28.0,14.0,14.0,14.0,14.0,14.0,28.0,12.0,6.0,2.0,6.0,8.0,8.0,8.0,10.0,12.0,16.0,18.0,18.0,14.0,10.0,10.0,2.0,-2.0,-4.0,0.0,14.0,14.0,14.0,-2.0,-2.0,-2.0,-2.0,-2.0,12.0,28.0,22.0,14.0,10.0,4.0,4.0,4.0,6.0,8.0,8.0,6.0,6.0,10.0,14.0,14.0,14.0,6.0,-4.0,-8.0,-2.0,-2.0,-2.0,-4.0,-4.0,-4.0,-4.0,-4.0,6.0,22.0,28.0,16.0,12.0,2.0,2.0,2.0,4.0,6.0,6.0,4.0,4.0,8.0,12.0,12.0,12.0,12.0,2.0,-2.0,-4.0,-4.0,-4.0,8.0,8.0,8.0,8.0,8.0,2.0,14.0,16.0,28.0,24.0,14.0,14.0,14.0,16.0,18.0,14.0,8.0,4.0,12.0,16.0,20.0,24.0,20.0,10.0,6.0,8.0,8.0,8.0,12.0,12.0,12.0,12.0,12.0,6.0,10.0,12.0,24.0,28.0,18.0,18.0,14.0,12.0,14.0,10.0,12.0,8.0,8.0,12.0,16.0,20.0,20.0,14.0,10.0,12.0,12.0,12.0,22.0,22.0,22.0,22.0,22.0,8.0,4.0,2.0,14.0,18.0,28.0,28.0,24.0,18.0,16.0,12.0,14.0,14.0,10.0,14.0,18.0,18.0,18.0,12.0,8.0,22.0,22.0,22.0,22.0,22.0,22.0,22.0,22.0,8.0,4.0,2.0,14.0,18.0,28.0,28.0,24.0,18.0,16.0,12.0,14.0,14.0,10.0,14.0,18.0,18.0,18.0,12.0,8.0,22.0,22.0,22.0,22.0,22.0,22.0,22.0,22.0,8.0,4.0,2.0,14.0,14.0,24.0,24.0,28.0,22.0,20.0,16.0,14.0,14.0,14.0,18.0,18.0,18.0,18.0,8.0,8.0,22.0,22.0,22.0,20.0,20.0,20.0,20.0,20.0,10.0,6.0,4.0,16.0,12.0,18.0,18.0,22.0,28.0,26.0,22.0,16.0,12.0,20.0,20.0,20.0,20.0,12.0,2.0,6.0,20.0,20.0,20.0,18.0,18.0,18.0,18.0,18.0,12.0,8.0,6.0,18.0,14.0,16.0,16.0,20.0,26.0,28.0,24.0,18.0,14.0,22.0,22.0,22.0,18.0,10.0,0.0,4.0,18.0,18.0,18.0,18.0,18.0,18.0,18.0,18.0,16.0,8.0,6.0,14.0,10.0,12.0,12.0,16.0,22.0,24.0,28.0,22.0,18.0,26.0,22.0,22.0,14.0,6.0,0.0,4.0,18.0,18.0,18.0,20.0,20.0,20.0,20.0,20.0,18.0,6.0,4.0,8.0,12.0,14.0,14.0,14.0,16.0,18.0,22.0,28.0,24.0,24.0,20.0,16.0,8.0,4.0,2.0,6.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,18.0,6.0,4.0,4.0,8.0,14.0,14.0,14.0,12.0,14.0,18.0,24.0,28.0,20.0,16.0,12.0,4.0,4.0,2.0,6.0,20.0,20.0,20.0,16.0,16.0,16.0,16.0,16.0,14.0,10.0,8.0,12.0,8.0,10.0,10.0,14.0,20.0,22.0,26.0,24.0,20.0,28.0,24.0,20.0,12.0,4.0,-2.0,2.0,16.0,16.0,16.0,12.0,12.0,12.0,12.0,12.0,10.0,14.0,12.0,16.0,12.0,14.0,14.0,18.0,20.0,22.0,22.0,20.0,16.0,24.0,28.0,24.0,16.0,8.0,-2.0,-2.0,12.0,12.0,12.0,12.0,12.0,12.0,12.0,12.0,10.0,14.0,12.0,20.0,16.0,18.0,18.0,18.0,20.0,22.0,22.0,16.0,12.0,20.0,24.0,28.0,20.0,12.0,2.0,-2.0,12.0,12.0,12.0,12.0,12.0,12.0,12.0,12.0,2.0,14.0,12.0,24.0,20.0,18.0,18.0,18.0,20.0,18.0,14.0,8.0,4.0,12.0,16.0,20.0,28.0,20.0,10.0,6.0,12.0,12.0,12.0,12.0,12.0,12.0,12.0,12.0,-2.0,6.0,12.0,20.0,20.0,18.0,18.0,18.0,12.0,10.0,6.0,4.0,4.0,4.0,8.0,12.0,20.0,28.0,18.0,14.0,12.0,12.0,12.0,10.0,10.0,10.0,10.0,10.0,-4.0,-4.0,2.0,10.0,14.0,12.0,12.0,8.0,2.0,0.0,0.0,2.0,2.0,-2.0,-2.0,2.0,10.0,18.0,28.0,24.0,10.0,10.0,10.0,14.0,14.0,14.0,14.0,14.0,0.0,-8.0,-2.0,6.0,10.0,8.0,8.0,8.0,6.0,4.0,4.0,6.0,6.0,2.0,-2.0,-2.0,6.0,14.0,24.0,28.0,14.0,14.0,14.0,28.0,28.0,28.0,28.0,28.0,14.0,-2.0,-4.0,8.0,12.0,22.0,22.0,22.0,20.0,18.0,18.0,20.0,20.0,16.0,12.0,12.0,12.0,12.0,10.0,14.0,28.0,28.0,28.0,28.0,28.0,28.0,28.0,28.0,14.0,-2.0,-4.0,8.0,12.0,22.0,22.0,22.0,20.0,18.0,18.0,20.0,20.0,16.0,12.0,12.0,12.0,12.0,10.0,14.0,28.0,28.0,28.0,28.0,28.0,28.0,28.0,28.0,14.0,-2.0,-4.0,8.0,12.0,22.0,22.0,22.0,20.0,18.0,18.0,20.0,20.0,16.0,12.0,12.0,12.0,12.0,10.0,14.0,28.0,28.0,28.0};
float float_in[784] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.011764707,0.070588239,0.070588239,0.070588239,0.49411768,0.53333336,0.68627453,0.10196079,0.65098041,1.0,0.96862751,0.49803925,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.11764707,0.14117648,0.36862746,0.60392159,0.66666669,0.99215692,0.99215692,0.99215692,0.99215692,0.99215692,0.88235301,0.67450982,0.99215692,0.94901967,0.76470596,0.25098041,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.19215688,0.9333334,0.99215692,0.99215692,0.99215692,0.99215692,0.99215692,0.99215692,0.99215692,0.99215692,0.98431379,0.36470589,0.32156864,0.32156864,0.21960786,0.15294118,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.070588239,0.8588236,0.99215692,0.99215692,0.99215692,0.99215692,0.99215692,0.77647066,0.71372551,0.96862751,0.9450981,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3137255,0.61176473,0.41960788,0.99215692,0.99215692,0.80392164,0.043137256,0.0,0.16862746,0.60392159,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.054901965,0.0039215689,0.60392159,0.99215692,0.35294119,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.54509807,0.99215692,0.74509805,0.0078431377,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.043137256,0.74509805,0.99215692,0.27450982,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.13725491,0.9450981,0.88235301,0.627451,0.42352945,0.0039215689,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.31764707,0.94117653,0.99215692,0.99215692,0.4666667,0.098039225,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.17647059,0.72941178,0.99215692,0.99215692,0.58823532,0.10588236,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.062745102,0.36470589,0.98823535,0.99215692,0.73333335,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.97647065,0.99215692,0.97647065,0.25098041,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.18039216,0.50980395,0.71764708,0.99215692,0.99215692,0.81176478,0.0078431377,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.15294118,0.58039218,0.89803928,0.99215692,0.99215692,0.99215692,0.98039222,0.71372551,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.094117656,0.44705886,0.86666673,0.99215692,0.99215692,0.99215692,0.99215692,0.78823537,0.30588236,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.090196081,0.25882354,0.83529419,0.99215692,0.99215692,0.99215692,0.99215692,0.77647066,0.31764707,0.0078431377,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.070588239,0.67058825,0.8588236,0.99215692,0.99215692,0.99215692,0.99215692,0.76470596,0.3137255,0.035294119,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.21568629,0.67450982,0.88627458,0.99215692,0.99215692,0.99215692,0.99215692,0.95686281,0.52156866,0.043137256,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.53333336,0.99215692,0.99215692,0.99215692,0.83137262,0.52941179,0.51764709,0.062745102,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
uint8_t W1[4] = {84,127,207,127};
float Bias1[2] = {0.010181473,0.0069620875};
float Gamma1[2] = {0.99622828,1.0009493};
float Beta1[2] = {0.094135918,-0.055339132};
float Mean1[2] = {-0.38179475,0.39702597};
float Std1[2] = {0.79487574,0.87844265};
uint8_t fdot_out[196] = {255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,254,0,127,248,0,15,254,0,3,255,240,0,191,255,1,31,255,232,15,255,255,193,223,255,254,31,255,255,208,127,255,254,129,255,255,244,15,255,255,224,127,255,253,7,255,255,192,127,255,240,7,255,252,1,255,255,128,55,255,192,13,255,240,3,255,255,0,255,255,255,219,255,254,131,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,128,0,127,240,0,127,255,0,15,255,240,0,255,224,0,15,252,0,0,94,96,0,1,224,0,0,15,128,0,0,254,0,0,7,240,0,0,63,128,0,0,248,0,0,15,128,0,3,248,0,0,255,0,0,63,248,0,15,254,0,3,255,128,0,127,224,0,15,252,0,0,127,0,0,0,0,0,0,0,0,0};
