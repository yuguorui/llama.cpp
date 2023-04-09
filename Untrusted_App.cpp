#include <cstddef>
#include <cstdio>
#include <cassert>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "sgx_urts.h"
#include "sgx_error.h"       /* sgx_status_t */
#include "sgx_eid.h"     /* sgx_enclave_id_t */

#include "Enclave_u.h"
# define ENCLAVE_FILENAME "llama_enclave.signed.so"

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;

int initialize_enclave(void)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    
    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        printf("Error, call sgx_create_enclave fail with 0x%x [%s].\n", ret, __FUNCTION__);
        return -1;
    }

    return 0;
}

void ocall_print_string(const char *str)
{
    printf("%s", str);
}

void ocall_print_string_stderr(const char *str)
{
    fprintf(stderr, "[Enclave] %s", str);
}

#define OCALL_DEFINE_0(name, ret_type)  \
    ret_type ocall_##name() {           \
        return name();                  \
    }

#define OCALL_DEFINE_1(name, ret_type, arg_type)    \
    ret_type ocall_##name(arg_type arg) {           \
        return name(arg);                           \
    }

#define OCALL_DEFINE_2(name, ret_type, arg_type1, arg_type2)    \
    ret_type ocall_##name(arg_type1 arg1, arg_type2 arg2) {     \
        return name(arg1, arg2);                                \
    }
#define OCALL_DEFINE_3(name, ret_type, arg_type1, arg_type2, arg_type3)     \
    ret_type ocall_##name(arg_type1 arg1, arg_type2 arg2, arg_type3 arg3) { \
        return name(arg1, arg2, arg3);                                      \
    }
#define OCALL_DEFINE_4(name, ret_type, arg_type1, arg_type2, arg_type3, arg_type4)          \
    ret_type ocall_##name(arg_type1 arg1, arg_type2 arg2, arg_type3 arg3, arg_type4 arg4) { \
        return name(arg1, arg2, arg3, arg4);                                                \
    }

OCALL_DEFINE_0(clock, clock_t)
OCALL_DEFINE_2(clock_gettime, int, clockid_t, struct timespec *)
OCALL_DEFINE_1(time, time_t, time_t *)
OCALL_DEFINE_2(open, int, const char *, int)
OCALL_DEFINE_3(lseek, off_t, int, off_t, int)
OCALL_DEFINE_3(read, ssize_t, int, void *, size_t)
OCALL_DEFINE_1(close, int, int)
OCALL_DEFINE_1(sleep, unsigned int, unsigned int)

/* Application entry */
int SGX_CDECL main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);

    int retval, ret;

    /* Initialize the enclave */
    if(initialize_enclave() < 0){
        return -1; 
    }
 
    ret = load_model(global_eid, &retval, "ggml-model-q4_0.bin", 1680686893);
    assert(ret == SGX_SUCCESS);
    printf("load_model return %d\n", retval);

    ret = completion(global_eid, &retval, "How to setup a website in 10 step: ", 6);
    assert(ret == SGX_SUCCESS);
    printf("completion return %d\n", retval);
    
    /* Destroy the enclave */
    sgx_destroy_enclave(global_eid);
    
    printf("Info: LLaMA.enclave successfully returned.\n");

    //printf("Enter a character before exit ...\n");
    //getchar();
    return 0;
}

