#ifndef _CL_HEAD_H_
#define _CL_HEAD_H_

#include <CL/cl.h>

void checkErr(cl_int err,const char *name);
cl_device_id ocl_init (char *target_name);
void contextQ(cl_device_id device_Id, cl_context *conteXt, cl_command_queue *queUe);
cl_program cProgram(const char*fileName, cl_context context, cl_device_id device_id);
cl_kernel cKernel(cl_program program, const char *kernel_name);
cl_kernel find_platform(char *name);
void find_CKQ(
    const char *platform_name, const char *kernel_file, const char *kernel_name,
    cl_context *context, cl_kernel *kernel, cl_command_queue *queue
);

#endif
