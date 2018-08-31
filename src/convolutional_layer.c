#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef CBLAS
#include <cblas.h>
#endif
#include "im2row.h"
#include "common_types.h"

#ifdef AI2
#include "xnor_layer.h"
#endif

#include "half.h"
void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

void binarize_w2sign(float *weights, int n, int size, unsigned int *sign, float *scale)
{
    int i, f;
    int word_index, bit_index;
    for(word_index = f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i)
            mean += fabs(weights[f*size + i]);
        scale[f] = mean / size;
        //printf("scale[%d]-%9.5f\n",f,scale[f]);fflush(stdout);
        for(bit_index = i = 0; i < size; ++i){
            if(weights[f*size + i]>0){
                sign[word_index] = ((sign[word_index]>>1)|0x80000000);
            }else{
                sign[word_index] = ((sign[word_index]>>1)&0x7fffffff);
            }
            //prbin(weights[f*size+i],sign[word_index]);
            if(bit_index++==31){
                //printf("%d word_index end\n",word_index);fflush(stdout);
                bit_index = 0;
                word_index++;
            }
        }
        if(bit_index!=0){
            sign[word_index]>>=bit_index;
            //printf("remain %d bit\n",bit_index);fflush(stdout);
            //prbin(0,sign[word_index]);
            word_index++;
        }else{
            //prbin(0,sign[word_index]);
        }
    }
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size); 
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif
    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &l->bf_algo);
}
#endif
#endif

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    float Bn=0;
    static float Total_Bn=0;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weight_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    l.nweights = c*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.biased_output = calloc(l.batch*l.outputs, sizeof(float));    //add
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

#ifdef OPENEXR
    l.weights_hf = (half*)calloc(c*n*size*size, sizeof(half));      //add
    l.biases_hf  = (half*)calloc(n, sizeof(half));                  //add
    l.output_hf  = (half*)calloc(l.batch*l.outputs, sizeof(half));  //add
    l.biased_output_hf = (half*)calloc(l.batch*l.outputs, sizeof(half));//add
#endif

    //l.forward = forward_convolutional_layer;  //remove original
    //l.forward = forward_convolutional_layer_cpu; //remove for test version
    //l.forward = forward_convolutional_layer_foldBN;    //add for fold batch normalize
    //l.forward = forward_convolutional_layer_kn2row;    //add for fold batch normalize
    l.forward = forward_convolutional_layer_hf;    //add for FPGA
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.cweights = calloc(c*n*size*size, sizeof(char));
        l.scales = calloc(n, sizeof(float));
        l.signWb = calloc(n*(int)ceilf(c*size*size/32.),sizeof(unsigned int));               //add
        l.signIb = calloc(c*size*size*(int)ceilf(out_w*out_h/32.),sizeof(unsigned int));     //add
        l.scale_alpha = calloc(n, sizeof(float));                           //add
        l.scale_beta  = calloc(c*size*size, sizeof(float));                 //add
        //printf("binary-weight=%d words signWb=%d words\n",c*n*size*size,(int)ceilf(c*size*size*n/32.));fflush(stdout);
    }
    if(xnor){
        l.binary_weights = calloc(c*n*size*size, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    l.done_norm = calloc(1,sizeof(unsigned int));   //add
    *l.done_norm = 0;                               //add

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));

#ifdef OPENEXR
        l.scales_hf           = (half*)calloc(n, sizeof(half)); //add
        l.rolling_mean_hf     = (half*)calloc(n, sizeof(half)); //add
        l.rolling_variance_hf = (half*)calloc(n, sizeof(half)); //add
#endif

    }
    if(adam){
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, c*n*size*size);
            l.v_gpu = cuda_make_array(l.v, c*n*size*size);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, c*n*size*size);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    //fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    Bn = (2.0 * l.n * l.size*l.size*l.c * l.out_h*l.out_w)/1000000000.;
    Total_Bn+= Bn;
    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f (%5.3f)BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, Bn, Total_Bn);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c*l.size*l.size; ++j){
            l.weights[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

// add_bias BLAS version
// But Precision btn add_bias() and add_bias_cblas() is difference, be carefull!
//
void add_bias_cblas(float *output, float *biases, int batch, int n, int size)   //add
{
#ifdef CBLAS
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            //for(j = 0; j < size; ++j){
            //    output[(b*n + i)*size + j] += biases[i];
            //}
            cblas_saxpy(size, 1, biases+i, 1, output+(b*n+i)*size, 1);
        }
    }
#endif
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void scale_add_biased_output(layer l){
    int size = l.out_h*l.out_w;
    int n    = l.out_c;
    int batch = l.batch;
    float *biased   = l.biased_output;
    float *biases   = l.biases;
    float *mean     = l.rolling_mean;
    float *variance = l.rolling_variance;
    float *scale    = l.scales;
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                biased[(b*n + i)*size + j] -= scale[i] * mean[i]/(sqrt(variance[i]) + .000001f);
                biased[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_add_bias(layer l, float *output){
    int size = l.out_h*l.out_w;
    int n    = l.out_c;
    int batch = l.batch;
    //float *output   = l.output;
    float *biases   = l.biases;
    float *mean     = l.rolling_mean;
    float *variance = l.rolling_variance;
    float *scale    = l.scales;
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] -= scale[i] * mean[i]/(sqrt(variance[i]) + .000001f);
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void normalize_weights(layer l, float *weights){
    int spatial = l.size*l.size*l.c;
    int filters = l.out_c;
    int batch   = l.batch;
    //float *w        = l.weights;
    float *variance = l.rolling_variance;
    float *scale    = l.scales;
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){ 
                int index = b*filters*spatial + f*spatial + i;
                weights[index] *= scale[f]/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void col2row_cblas(int sz_col, int sz_row, float* colm_src, float* rowm_dst){
#ifdef CBLAS
    int r;
    for(r=0; r<sz_row; r++)
        cblas_scopy(sz_col, colm_src+r, sz_row, rowm_dst+r*sz_col, 1);
#endif
}

void row2col_major(int sz_col, int sz_row, float *rowm_src, float *rowm_dst){
    int c,r;
    int m,n;
    for(r=0;r<sz_row;r++)
        for(c=0;c<sz_col;c++){
            m = r*sz_col + c;
            n = c*sz_row + r;
            rowm_dst[n] = rowm_src[m];
        }
}

void col2row_major(int sz_col, int sz_row, float *colm_src, float *rowm_dst){
    int c,r;
    int m,n;
    for(c=0;c<sz_col;c++)
        for(r=0;r<sz_row;r++){
            m = r*sz_col + c;
            n = c*sz_row + r;
            rowm_dst[m] = colm_src[n];
        }
}

void expand_line_length(int lines, int line_length, int expand_length, float*X, float*Y){
    int i,j;
    for(j=0; j<lines; j++)
        for(i=0; i<line_length + expand_length; i++)
            if(i<line_length)
                Y[j*(line_length + expand_length) + i] =X[j*line_length + i];
            else
                Y[j*(line_length + expand_length) + i] =0;
}

void dump_weights_3_16(int index, int m, int n, int k, float *weights){
    int i,j;
    int gentypen = (!(k%16))?16:3;
    char file_name[128];
    sprintf(file_name,"weights_%d.cl",index);
    FILE *fp=fopen(file_name,"w");
    fprintf(fp,"constant float%d weights_layer_%d_row_major[%d*%d] = {\n",gentypen,index,m,k);
    for(i=0;i<m*k;i+=gentypen){
        fprintf(fp,"{");
        for(j=0;j<gentypen;j++){
            fprintf(fp,"%f",weights[i+j]);
            if(j<gentypen-1) fprintf(fp,",");
        }
        fprintf(fp,"}");
        if(i<m*k-gentypen-1) fprintf(fp,",\n");
    }
    fprintf(fp,"\n};\n");
    fclose(fp);
}

void forward_convolutional_layer_hf(convolutional_layer l, network net)
{
    int i;
    int out_h = l.out_h;
    int out_w = l.out_w;

#ifdef FPGA
    if(!get_FPGA_init()){set_FPGA_init();gemm_fpga_init();}
#endif
    double time=what_time_is_it_now();

    //copy_cpu(l.outputs*l.batch, l.biased_output, 1, l.output, 1);
#ifdef CBLAS
    cblas_scopy(l.outputs*l.batch, l.biased_output, 1, l.output, 1);
#endif
#ifdef OPENEXR
    for(i=0;i<l.outputs*l.batch;i++) l.output_hf[i]=l.biased_output[i];
#endif

    // with im2col version for gemm_nn
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;
    if(0){
        float *a = l.weights;
        float *b = net.workspace;
        float *c = l.output;

        im2col_cpu(net.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
        printf("%9.6f ", what_time_is_it_now()-time);
        gemm2(0, 0, 0, m, n, k, 1, a, k, b, n, 1, c, n);    //OK FPGA gemm1_naive.aocx
    }else if(0){
#ifdef OPENEXR
        float *b = net.workspace;
        float *c = l.output;
        half *a_hf = l.weights_hf;
        half *b_hf = net.workspace_hf;
        half *c_hf = l.output_hf;

        im2col_cpu(net.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
        float2half(k*n, b, 1, b_hf, 1);
#ifdef GET_LAYER_TIME
        printf("%9.6f ", what_time_is_it_now()-time);
#endif
        gemm_hf(0, 0, 0, m, n, k, 1, a_hf, k, b_hf, n, 1, c_hf, n);    //OK FPGA gemm1_naive_half.aocx
        half2float(m*n, c_hf, 1, c, 1);
#else
        error("Need OPENEXR Define-0");
#endif
    }else if(0){    // for gemm1_halfxf_halfx9.cl im2col+row2col_major
#ifdef OPENEXR
        float *b = net.workspace;
        float *c = l.output;
        half *a_hf = l.weights_hf;
        half *b_hf = net.workspace_hf;
        half *c_hf = l.output_hf;

        float *B = (float*)malloc(sizeof(float)*k*n);
        int k_pad = ((k%16))?((int)(k/16+1)*16)%k:0;
        float *b_pad = (float*)malloc(sizeof(float)*(k+k_pad)*n);

        //dump_weights_3_16(net.index, m, n, k, l.weights);
        im2col_cpu(net.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
        row2col_major(n, k, b, B);

        //expand_line_length(n, k, k_pad, B, b_pad);
        //float2half((k+k_pad)*n, b_pad, 1, b_hf, 1);
        float2half(k*n, B, 1, b_hf, 1);
        printf("%9.6f ", what_time_is_it_now()-time);
        //gemm_hf(0, 1, 0, m, n, k, 1, a_hf, k, b_hf, k+k_pad, 1, c_hf, n);
        gemm_hf(0, 1, 0, m, n, k, 1, a_hf, k, b_hf, k, 1, c_hf, n);
        half2float(m*n, c_hf, 1, c, 1);
        free(B);free(b_pad);
#else
        error("Need OPENEXR Define-0");
#endif
    }

    // with im2row version for gemm_ntt
    m = out_h*out_w;
    k = l.size*l.size*l.c;
    n = l.n;
    if(0){ // with FPGA Model for gemm_ntt_float.cl
        float *a = net.workspace;
        float *b = l.weights;
        float *c = l.output;
        float *A = (float*)malloc(sizeof(float)*(l.out_w*l.out_h)*(l.size*l.size*l.c));
        TensorDim in_dim  ={ 1, l.c, l.h, l.w };
        TensorDim filt_dim={ l.out_c, l.c, l.size, l.size };
        CppConvnetIm2Row(a, net.input, out_w, out_h, k, in_dim, filt_dim, l.stride, l.pad);
        //col2row_cblas(l.c*l.size*l.size, out_w*out_h, a, A);
        col2row_major(l.c*l.size*l.size, out_w*out_h, a, A);
        //col2row_major(k,m,b,B);
        //row2col_major(l.c*l.size*l.size, out_w*out_h, A, a);
        printf("%9.6f ", what_time_is_it_now()-time);

        //gemm2(1,1,1, m, n, k, 1, a, m, b, k, 1, c, m);     //OK for instead of FPGA Model
        gemm2(0,1,1, m, n, k, 1, A, k, b, k, 1, c, m);     //OK for instead of FPGA Model
        free(A);
    }else if(0){ // All OpenBLAS
            float *a = net.workspace;
            float *b = l.weights;
            float *c = l.output;
            TensorDim in_dim  ={ 1, l.c, l.h, l.w };
            TensorDim filt_dim={ l.out_c, l.c, l.size, l.size };
            CppConvnetIm2Row(a, net.input, out_w, out_h, k, in_dim, filt_dim, l.stride, l.pad);
       //     printf("%9.6f ", what_time_is_it_now()-time);
#ifdef CBLAS
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, m, b, k, 1, c, m); //OK
#endif
    }else if(1){ // with FPGA Model for gemm_ntt.cl and gemm_ntt_jik.cl and gemm_ntt_jikK.cl
#ifdef OPENEXR
        //if(net.index==0 || net.index==2){
        if(1 && (net.index==0 || net.index==2 || net.index==7)){
            float *a = net.workspace;
            float *b = l.weights;
            float *c = l.output;
            TensorDim in_dim  ={ 1, l.c, l.h, l.w };
            TensorDim filt_dim={ l.out_c, l.c, l.size, l.size };
            CppConvnetIm2Row(a, net.input, out_w, out_h, k, in_dim, filt_dim, l.stride, l.pad);
            printf("%9.6f ", what_time_is_it_now()-time);
#ifdef CBLAS
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, m, b, k, 1, c, m); //OK
#endif
        }else if(0){    // run with budget btwn cblas and gemm_ntt_jikK.cl
            int N1 = n/16, N2 = n-N1;
            float *a = net.workspace;
            float *b = l.weights;
            float *c = l.output;
            float *A = (float*)malloc(sizeof(float)*(l.out_w*l.out_h)*(l.size*l.size*l.c));
            half *a_hf = net.workspace_hf;
            half *b_hf = l.weights_hf;
            half *c_hf = l.output_hf;
            TensorDim in_dim  ={ 1, l.c, l.h, l.w };
            TensorDim filt_dim={ l.out_c, l.c, l.size, l.size };
            CppConvnetIm2Row(a, net.input, out_w, out_h, k, in_dim, filt_dim, l.stride, l.pad);
            col2row_cblas(l.c*l.size*l.size, out_w*out_h, a, A);
            float2half(m*k, A, 1, a_hf, 1);
            printf("%9.6f ", what_time_is_it_now()-time);
            set_Nonblocking_launch();
            gemm_hf(0,1,1, m, N1, k, 1, a_hf, k, b_hf, k, 1, c_hf, m);     //OK for instead of FPGA Model
#ifdef CBLAS
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, N2, k, 1, a, m, b+N1*k, k, 1, c+N1*m, m); //OK
#endif
            wait_kernel_finish();
            half2float(m*N1, c_hf, 1, c, 1);
            free(A);
        }else{          // gemm_ntt_jikK.cl
            float *a = net.workspace;
            //float *b = l.weights;
            float *c = l.output;
            float *A = (float*)malloc(sizeof(float)*(l.out_w*l.out_h)*(l.size*l.size*l.c));
            half *a_hf = net.workspace_hf;
            half *b_hf = l.weights_hf;
            half *c_hf = l.output_hf;
            TensorDim in_dim  ={ 1, l.c, l.h, l.w };
            TensorDim filt_dim={ l.out_c, l.c, l.size, l.size };
            CppConvnetIm2Row(a, net.input, out_w, out_h, k, in_dim, filt_dim, l.stride, l.pad);
            col2row_cblas(l.c*l.size*l.size, out_w*out_h, a, A);
            //col2row_major(l.c*l.size*l.size, out_w*out_h, a, A);
            //col2row_major(l.c*l.size*l.size, m, a, A);
            //col2row_major(k,m,b,B);
            //row2col_major(l.c*l.size*l.size, out_w*out_h, A, a);

            float2half(m*k, A, 1, a_hf, 1);
            printf("%9.6f ", what_time_is_it_now()-time);
            gemm_hf(0,1,1, m, n, k, 1, a_hf, k, b_hf, k, 1, c_hf, m);     //OK for instead of FPGA Model
            half2float(m*n, c_hf, 1, c, 1);
            free(A);
        }
#else
            error("Need OPENEXR Define-1");
#endif
    }

    if(!l.batch_normalize){
        //add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
        add_bias_cblas(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

    activate_array(l.output, m*n*l.batch, l.activation);
}

void forward_convolutional_layer_foldBN(convolutional_layer l, network net)
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    double time=what_time_is_it_now();

    //copy_cpu(l.outputs*l.batch, l.biased_output, 1, l.output, 1);
#ifdef CBLAS
    cblas_scopy(l.outputs*l.batch, l.biased_output, 1, l.output, 1);
#endif

    // with im2col version
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;
    if(0){
        float *a = l.weights;
        float *b = net.workspace;
        float *c = l.output;

        im2col_cpu(net.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
        printf("%9.6f ", what_time_is_it_now()-time);
#ifdef CBLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 1, c, n); //OK
#endif
        //gemm2(0, 0, 0, m, n, k, 1, a, k, b, n, 1, c, n);    //OK
    }else if(0){ // with FPGA Model
        float *a = l.weights;
        float *b = net.workspace;
        float *c = l.output;

        im2col_cpu_col_major(net.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
        printf("%9.6f ", what_time_is_it_now()-time);
        gemm2(0, 1, 0, m, n, k, 1, a, k, b, k, 1, c, n);    //OK for instead of FPGA Model
    }

    // with im2row version
    m = out_h*out_w;
    k = l.size*l.size*l.c;
    n = l.n;
    if(1){
        float *a = net.workspace;
        float *b = l.weights;
        float *c = l.output;
        TensorDim in_dim  ={ 1, l.c, l.h, l.w };
        TensorDim filt_dim={ l.out_c, l.c, l.size, l.size };
        CppConvnetIm2Row(a, net.input, out_w, out_h, k, in_dim, filt_dim, l.stride, l.pad);
        printf("%9.6f ", what_time_is_it_now()-time);
#ifdef CBLAS
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, m, b, k, 1, c, m); //OK
#endif
    }else if(0){ // with FPGA Model
        float *a = net.workspace;
        float *b = l.weights;
        float *c = l.output;
        float *A = (float*)malloc(sizeof(float)*(l.out_w*l.out_h)*(l.size*l.size*l.c));
        //float *B = (float*)malloc(sizeof(float)*k*m);
        TensorDim in_dim  ={ 1, l.c, l.h, l.w };
        TensorDim filt_dim={ l.out_c, l.c, l.size, l.size };
        CppConvnetIm2Row(a, net.input, out_w, out_h, k, in_dim, filt_dim, l.stride, l.pad);
        double time=what_time_is_it_now();
        //col2row_cblas(l.c*l.size*l.size, out_w*out_h, a, A);
        col2row_major(l.c*l.size*l.size, out_w*out_h, a, A);
        //col2row_major(k,m,b,B);
        //row2col_major(l.c*l.size*l.size, out_w*out_h, A, a);
        printf("%9.6f ", what_time_is_it_now()-time);
        gemm2(0,1,1, m, n, k, 1, A, k, b, k, 1, c, m);     //OK for instead of FPGA Model
        free(A);
    }else if(0){
        float *a = net.workspace;
        float *b = l.weights;
        float *c = l.output;
        TensorDim in_dim  ={ 1, l.c, l.h, l.w };
        TensorDim filt_dim={ l.out_c, l.c, l.size, l.size };
        CppConvnetIm2Row(a, net.input, out_w, out_h, k, in_dim, filt_dim, l.stride, l.pad);
        double time=what_time_is_it_now();
        printf("%9.6f ", what_time_is_it_now()-time);
        gemm2(1,1,1, m, n, k, 1, a, m, b, k, 1, c, m);     //OK BLAS Spec
    }

    if(!l.batch_normalize){
        //add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
        add_bias_cblas(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

    activate_array(l.output, m*n*l.batch, l.activation);
}

void forward_convolutional_layer_cpu(convolutional_layer l, network net)
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    int i;
    //int pre_norm=0; //post-normalize-weights post-scale-biase post-add-biase for Training
    //int pre_norm=1; //pre-normalize-weights  post-biases for Only Prediction
    //int pre_norm=2; //pre-normalize-weights  pre-biases  for Only Prediction
    //int pre_norm=3; //normalization at load_weights      for Only Prediction
    int pre_norm=2;
    double time=what_time_is_it_now();
#ifdef FOLDBN
    pre_norm=3;
    copy_cpu(l.outputs*l.batch, l.biased_output, 1, l.output, 1);
#endif
    if(net.train) pre_norm=0;

    if((pre_norm<=1)||(pre_norm==2 && !*l.done_norm))
        fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    if(l.batch_normalize){
        if(pre_norm==2){
            if(!*l.done_norm)
                scale_add_bias(l, l.biased_output);
            copy_cpu(l.outputs*l.batch, l.biased_output, 1, l.output, 1);
        }
    }
    if(l.binary){
        binarize_w2sign(l.weights, l.n, l.c*l.size*l.size, l.signWb, l.scale_alpha);
//        binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        if(l.x_mean==0)
            binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        else
            for(i = 0; i < l.batch; ++i){
                binarize_input(net.input + i*l.inputs, l.c, l.h*l.w, l.binary_input + i*l.inputs);
            }
        net.input = l.binary_input;
    }

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;


    float *a = l.weights;
    float *b = net.workspace;
    float *c = l.output;

    if((pre_norm==1 || pre_norm==2) && !*l.done_norm && l.batch_normalize)
        normalize_weights(l, l.weights);

    for(i = 0; i < l.batch; ++i){
        im2col_cpu(net.input, l.c, l.h, l.w, 
               l.size, l.stride, l.pad, b);
        //im2col_cpu_col_major(net.input, l.c, l.h, l.w, 
        //        l.size, l.stride, l.pad, b);
        printf(" WOG=%f ", what_time_is_it_now()-time);
        if(l.binary)
            gemm_nn_sign(m,n,k,l.scale_alpha,l.signWb,k,b,n,c,n);
            //gemm_nn_binary(m,n,k,a,k,b,n,c,n);
        else
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        c += n*m;
        net.input += l.c*l.h*l.w;
    }

    if(l.batch_normalize){
        if(pre_norm==1)
            scale_add_bias(l, l.output);
        if(pre_norm==0)
            forward_batchnorm_layer(l, net);
        *l.done_norm = 1;
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

    activate_array(l.output, m*n*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

void forward_convolutional_layer(convolutional_layer l, network net)
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    if(l.binary){
        binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        if(l.x_mean==0)
            binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        else
            for(i = 0; i < l.batch; ++i){
                binarize_input(net.input + i*l.inputs, l.c, l.h*l.w, l.binary_input + i*l.inputs);
            }
        net.input = l.binary_input;
    }

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;


    float *a = l.weights;
    float *b = net.workspace;
    float *c = l.output;

    for(i = 0; i < l.batch; ++i){
        im2col_cpu(net.input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        c += n*m;
        net.input += l.c*l.h*l.w;
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

    activate_array(l.output, m*n*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i;
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        float *a = l.delta + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates;

        float *im = net.input+i*l.c*l.h*l.w;

        im2col_cpu(im, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(net.delta){
            a = l.weights;
            b = l.delta + i*m*k;
            c = net.workspace;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            col2im_cpu(net.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, net.delta+i*l.c*l.h*l.w);
        }
    }
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
        char buff[256];
        sprintf(buff, "filter%d", i);
        save_image(weights[i], buff);
        */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

