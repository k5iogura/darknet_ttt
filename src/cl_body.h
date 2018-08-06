#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifndef _CL_BODY_H_
#define _CL_BODY_H_

#include <CL/cl.h>

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x500000)

#define MAX_ENV (10)
static cl_device_id device_id[MAX_ENV];
static cl_platform_id platform_id[MAX_ENV];

void checkErr(cl_int err,const char *name)
{
	if(err != CL_SUCCESS)
	{
		printf("Error: %s %d",name,err);
		switch(err)
		{
			case CL_DEVICE_NOT_FOUND :printf("(CL_DEVICE_NOT_FOUND)");break;
			case CL_DEVICE_NOT_AVAILABLE :printf("(CL_DEVICE_NOT_AVAILABLE)");break;
			case CL_COMPILER_NOT_AVAILABLE :printf("(CL_COMPILER_NOT_AVAILABLE)");break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE :printf("(CL_MEM_OBJECT_ALIOCATION_FAILURE)");break;
			case CL_OUT_OF_RESOURCES :printf("(CL_OUT_OF_RESOURCES)");break;
			case CL_OUT_OF_HOST_MEMORY :printf("(CL_OUT_OF_HOST_MEMORY)");break;
			case CL_MEM_COPY_OVERLAP :printf("(CL_MEM_COPY_OVERLAP)");break;
			case CL_BUILD_PROGRAM_FAILURE:printf("(CL_BUILD_PROGRAM_FAILURE)");break;
			case CL_INVALID_VALUE:printf("(CL_INVALID_VALUE)");break;
			case CL_INVALID_DEVICE_TYPE:printf("(CL_INVALID_DEVICE_TYPE)");break;
			case CL_INVALID_DEVICE:printf("(CL_INVALID_DEVICE)");break;
			case CL_INVALID_CONTEXT:printf("(CL_INVALID_CONTEXT)");break;
			case CL_INVALID_BINARY:printf("(CL_INVALID_BINARY)");break;
			case CL_INVALID_BUILD_OPTIONS:printf("(CL_INVALID_BUILD_OPTIONS)");break;
			case CL_INVALID_PROGRAM:printf("(CL_INVALID_PROGRAM)");break;
			case CL_INVALID_PROGRAM_EXECUTABLE:printf("(CL_INVALID_PROGRAM_EXECUTABLE)");break;
			case CL_INVALID_KERNEL_DEFINITION:printf("(CL_INVALID_KERNEL_DEFINITION)");break;
			case CL_INVALID_KERNEL:printf("(CL_INVALID_KERNEL)");break;
			case CL_INVALID_KERNEL_ARGS:printf("(CL_INVALID_KERNEL_ARGS)");break;
			case CL_INVALID_OPERATION:printf("(CL_INVALID_OPERATION)");break;
			case CL_INVALID_COMMAND_QUEUE:printf("(CL_INVALID_COMMAND_QUEUE)");break;
			case CL_INVALID_WORK_DIMENSION:printf("(CL_INVALID_WORK_DIMENSION)");break;
			case CL_INVALID_WORK_GROUP_SIZE:printf("(CL_INVALID_WORK_GROUP_SIZE)");break;
			case CL_INVALID_WORK_ITEM_SIZE:printf("(CL_INVALID_WORK_ITEM_SIZE)");break;
			case CL_INVALID_GLOBAL_WORK_SIZE:printf("(CL_INVALID_GLOBAL_WORK_SIZE)");break;
			case CL_INVALID_GLOBAL_OFFSET:printf("(CL_INVALID_GLOBAL_OFFSET)");break;
			case CL_INVALID_IMAGE_SIZE:printf("(CL_INVALID_IMAGE_SIZE)");break;
			case CL_INVALID_EVENT_WAIT_LIST:printf("(CL_INVALID_EVENT_WAIT_LIST)");break;
			case CL_MISALIGNED_SUB_BUFFER_OFFSET:printf("(CL_MISALIGNED_SUB_BUFFER_OFFSET)");break;

			default:
												 break;
		}
		printf("\n");
		exit(1);
	}
}

cl_device_id ocl_init (const char *target_name) {
    int i,j,k;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    cl_int ret1,ret2,ret3;
    cl_platform_id target_platform = NULL;
    cl_device_id   target_device = NULL;

    /* Load the source code containing the kernel*/

    /* Get Platform */
    ret = clGetPlatformIDs (MAX_ENV, platform_id, &ret_num_platforms);
    checkErr(ret,"clGetPlatformIDs");
    printf("%d platforms\n",ret_num_platforms);

    /* Get Device and info */
    cl_ulong local_mem;
    char platform_name[1024], device_name[1024];
    for(i=0;i<(int)ret_num_platforms;i++){
        clGetPlatformInfo(platform_id[i],CL_PLATFORM_NAME,sizeof(platform_name),platform_name,NULL);
        printf("\tNo.%d-\"%s\"\n",i,platform_name);
        ret = clGetDeviceIDs (platform_id[i], CL_DEVICE_TYPE_DEFAULT, 1, device_id, &ret_num_devices);
        checkErr(ret,"clGetDeviceIds");
        printf("\t%d devices\n",ret_num_devices);
        for(j=0;j<(int)ret_num_devices;j++){
            clGetDeviceInfo(device_id[j], CL_DEVICE_NAME,sizeof(device_name),device_name,NULL);
            clGetDeviceInfo(device_id[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),&local_mem,NULL);
            printf("\t\tNo.%d-\"%s\" : LOCAL_MEM_SIZE=%lu\n",j,device_name,local_mem);
        }
        if(target_name && !strcmp(platform_name, target_name) && ret_num_devices>0) target_device = device_id[0];
    }
    return target_device;
}

void contextQ(cl_device_id device_Id, cl_context *conteXt, cl_command_queue *queUe){
    cl_int ret;
    /* Create OpenCL context */
    *conteXt = clCreateContext (NULL, 1, &device_Id, NULL, NULL, &ret);
    checkErr(ret,"clCreateContext");

    /* Create Command Queue */
    *queUe = clCreateCommandQueue (*conteXt, device_Id, 0, &ret);
    checkErr(ret,"clCreateCommandQueue");
}

cl_program cProgram(const char*fileName, cl_context context, cl_device_id device_id){
    const unsigned char *source_str;
    size_t source_size;
    cl_program program = NULL;
    cl_int ret;

    /* Load the source code containing the kernel*/
    FILE *fp = fopen (fileName, "r");
    if (!fp) {
        fprintf (stderr, "Failed to load kernel.\n");
        exit (1);
    }
    source_str = (const unsigned char *) malloc (MAX_SOURCE_SIZE);
    source_size = fread ((void*)source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose (fp);

    /* Create Kernel Program from the source */
    #ifdef onX86
        program = clCreateProgramWithSource (context, 1, (const char **) &source_str,
                                   (const size_t *) &source_size, &ret);
        checkErr(ret,"clCreateProgramWithSource");
    #else
        cl_int ret1;
        program = clCreateProgramWithBinary (context, 1, &device_id, &source_size,
            (const unsigned char **) &source_str, &ret1, &ret);
        checkErr(ret,"clCreateProgramWithBinary");
    #endif

    /* Build Kernel Program */
    ret = clBuildProgram (program, 1, &device_id, NULL, NULL, NULL);
    checkErr(ret,"clBuildProgram");

    cl_kernel kernels[MAX_ENV];
    cl_uint   n_kernels;
    ret = clCreateKernelsInProgram(program,2,kernels,&n_kernels);
    checkErr(ret,"clCreateKernelsInProgram");

    char name[64];
    size_t info_size;
    for(int i=0;i<(int)n_kernels;i++){
        (void)clGetKernelInfo(kernels[i],CL_KERNEL_FUNCTION_NAME,64,name,&info_size);
        printf("In Program kernel[%d] name = %s\n",i,name);
    //    clReleaseKernel(kernels[i]);
    }
    return program;
}
cl_kernel cKernel(cl_program program, const char *kernel_name){
    cl_kernel  kernel = NULL;
    cl_int ret;
    /* Create OpenCL Kernel */
    //char kernel_name[128]="gemm_nn";
    //char kernel_name[128]="gemm_nnA1";
    //char kernel_name[128]="gemm_nnWB";
    //char kernel_name[128]="gemm_nn4W";
    kernel = clCreateKernel (program, kernel_name, &ret);
    checkErr(ret,"clCreateKernel");
    return kernel;
}

void find_CKQ(
    const char *platform_name, const char *kernel_file, const char *kernel_name,
    cl_context *context, cl_kernel *kernel, cl_command_queue *queue
){
    cl_device_id device_id = ocl_init(platform_name);
    cl_program program=NULL;
    if(device_id)
        contextQ(device_id, context, queue);
    if(*context)
        program = cProgram(kernel_file,*context,device_id);
    if(program)
        *kernel = cKernel(program,kernel_name);
}

void find_CnKQ(
    const char *platform_name, const char *kernel_file, int n, const char *kernel_name[],
    cl_context *context, cl_kernel kernel[], cl_command_queue *queue
){
    int i;
    cl_device_id device_id = ocl_init(platform_name);
    cl_program program=NULL;
    if(device_id)
        contextQ(device_id, context, queue);
    if(*context)
        program = cProgram(kernel_file,*context,device_id);
    if(program)
        for(i=0;i<n;i++)
            kernel[i] = cKernel(program,kernel_name[i]);
}

cl_kernel find_platform(char *name){
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel  kernel;

    cl_device_id device_id = ocl_init(name);
    if(device_id)
        contextQ(device_id, &context, &queue);
    if(context)
        program = cProgram("gemm1.aocx",context,device_id);
    if(program)
        kernel = cKernel(program,"gemm_nn4W");
    return kernel;
}

#endif
