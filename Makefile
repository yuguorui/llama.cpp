#
# Compile flags
#
#

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686))
	ifeq ($(UNAME_S),Linux)
		AVX1_M := $(shell grep "avx " /proc/cpuinfo)
		ifneq (,$(findstring avx,$(AVX1_M)))
			CFLAGS += -mavx
		endif
		AVX2_M := $(shell grep "avx2 " /proc/cpuinfo)
		ifneq (,$(findstring avx2,$(AVX2_M)))
			CFLAGS += -mavx2
		endif
		FMA_M := $(shell grep "fma " /proc/cpuinfo)
		ifneq (,$(findstring fma,$(FMA_M)))
			CFLAGS += -mfma
		endif
		F16C_M := $(shell grep "f16c " /proc/cpuinfo)
		ifneq (,$(findstring f16c,$(F16C_M)))
			CFLAGS += -mf16c
		endif
		SSE3_M := $(shell grep "sse3 " /proc/cpuinfo)
		ifneq (,$(findstring sse3,$(SSE3_M)))
			CFLAGS += -msse3
		endif
		AVX512F_M := $(shell grep "avx512f " /proc/cpuinfo)
		ifneq (,$(findstring avx512f,$(AVX512F_M)))
			CFLAGS += -mavx512f
		endif
		AVX512BW_M := $(shell grep "avx512bw " /proc/cpuinfo)
		ifneq (,$(findstring avx512bw,$(AVX512BW_M)))
			CFLAGS += -mavx512bw
		endif
		AVX512DQ_M := $(shell grep "avx512dq " /proc/cpuinfo)
		ifneq (,$(findstring avx512dq,$(AVX512DQ_M)))
			CFLAGS += -mavx512dq
		endif
		AVX512VL_M := $(shell grep "avx512vl " /proc/cpuinfo)
		ifneq (,$(findstring avx512vl,$(AVX512VL_M)))
			CFLAGS += -mavx512vl
		endif
		AVX512CD_M := $(shell grep "avx512cd " /proc/cpuinfo)
		ifneq (,$(findstring avx512cd,$(AVX512CD_M)))
			CFLAGS += -mavx512cd
		endif
		AVX512ER_M := $(shell grep "avx512er " /proc/cpuinfo)
		ifneq (,$(findstring avx512er,$(AVX512ER_M)))
			CFLAGS += -mavx512er
		endif
		AVX512IFMA_M := $(shell grep "avx512ifma " /proc/cpuinfo)
		ifneq (,$(findstring avx512ifma,$(AVX512IFMA_M)))
			CFLAGS += -mavx512ifma
		endif
		AVX512PF_M := $(shell grep "avx512pf " /proc/cpuinfo)
		ifneq (,$(findstring avx512pf,$(AVX512PF_M)))
			CFLAGS += -mavx512pf
		endif
		CFLAGS += -mfma -mf16c -mavx -mavx2
	endif
endif

######## SGX related configuration ########
SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= HW
SGX_DEBUG ?= 1

include $(SGX_SDK)/buildenv.mk

SGX_COMMON_FLAGS := -m64
SGX_LIBRARY_PATH := $(SGX_SDK)/lib64
SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x64/sgx_sign
SGX_EDGER8R := $(SGX_SDK)/bin/x64/sgx_edger8r

ifeq ($(SGX_DEBUG), 1)
ifeq ($(SGX_PRERELEASE), 1)
$(error Cannot set SGX_DEBUG and SGX_PRERELEASE at the same time!!)
endif
endif

ifeq ($(SGX_DEBUG), 1)
        SGX_COMMON_FLAGS += -O0 -g
else
        SGX_COMMON_FLAGS += -O3 
endif

SGX_COMMON_FLAGS += $(CFLAGS) -Wall -Wextra -Winit-self -Wpointer-arith -Wreturn-type \
                    -Waddress -Wsequence-point -Wformat-security \
                    -Wmissing-include-dirs -Wno-unused-variable -Wno-unused-function -Wno-unused-parameter # -Wfloat-equal -Wundef -Wshadow \
                    -Wcast-align -Wcast-qual -Wconversion -Wredundant-decls
SGX_COMMON_CFLAGS := $(SGX_COMMON_FLAGS) -std=gnu11 -Wjump-misses-init # -Wstrict-prototypes -Wunsuffixed-float-constants
SGX_COMMON_CXXFLAGS := $(SGX_COMMON_FLAGS) -Wnon-virtual-dtor -std=c++11

ifneq ($(SGX_MODE), HW)
	Urts_Library_Name := sgx_urts_sim
else
	Urts_Library_Name := sgx_urts
endif

ifneq ($(SGX_MODE), HW)
	Trts_Library_Name := sgx_trts_sim
	Service_Library_Name := sgx_tservice_sim
else
	Trts_Library_Name := sgx_trts
	Service_Library_Name := sgx_tservice
endif
Crypto_Library_Name := sgx_tcrypto

######## App Settings ########
App_Include_Paths := -I. -I$(SGX_SDK)/include
App_C_Flags := -fPIC -Wno-attributes $(App_Include_Paths)
ifeq ($(SGX_DEBUG), 1)
        App_C_Flags += -DDEBUG -UNDEBUG -UEDEBUG
else ifeq ($(SGX_PRERELEASE), 1)
        App_C_Flags += -DNDEBUG -DEDEBUG -UDEBUG
else
        App_C_Flags += -DNDEBUG -UEDEBUG -UDEBUG
endif

App_Cpp_Flags := $(App_C_Flags)
App_Link_Flags := -L$(SGX_LIBRARY_PATH) -l$(Urts_Library_Name) -lpthread 

######## Enclave Settings ########
Enclave_Include_Paths := -I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc -I$(SGX_SDK)/include/libcxx -I/usr/lib/gcc/x86_64-linux-gnu/11/include -I/usr/lib/gcc/x86_64-redhat-linux/10/include/

Enclave_C_Flags := $(Enclave_Include_Paths) -nostdinc -fvisibility=hidden -fpie -ffunction-sections -fdata-sections $(MITIGATION_CFLAGS) -D__SGX_ENCLAVE__
Enclave_C_Flags += -fstack-protector-strong
Enclave_Cpp_Flags := $(Enclave_C_Flags) -nostdinc++
Enclave_Security_Link_Flags := -Wl,-z,relro,-z,now,-z,noexecstack
Enclave_Link_Flags := $(MITIGATION_LDFLAGS) $(Enclave_Security_Link_Flags) \
    -Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L$(SGX_TRUSTED_LIBRARY_PATH) \
	-Wl,--whole-archive -l$(Trts_Library_Name) -Wl,--no-whole-archive \
	-Wl,--start-group -lsgx_tstdc -lsgx_tcxx -l$(Crypto_Library_Name) -l$(Service_Library_Name) -lsgx_pthread -Wl,--end-group \
	-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
	-Wl,-pie,-eenclave_entry -Wl,--export-dynamic  \
	-Wl,--defsym,__ImageBase=0 -Wl,--gc-sections   \
	-Wl,--version-script=Enclave.lds


ifeq ($(SGX_MODE), HW)
ifeq ($(SGX_DEBUG), 1)
	Build_Mode = HW_DEBUG
else ifeq ($(SGX_PRERELEASE), 1)
	Build_Mode = HW_PRERELEASE
else
	Build_Mode = HW_RELEASE
endif
else
ifeq ($(SGX_DEBUG), 1)
	Build_Mode = SIM_DEBUG
else ifeq ($(SGX_PRERELEASE), 1)
	Build_Mode = SIM_PRERELEASE
else
	Build_Mode = SIM_RELEASE
endif
endif


#
# Print build information
#

$(info I llama.cpp build info: )
$(info I Enclave_C_Flags:	$(Enclave_C_Flags))
$(info I Enclave_Cpp_Flags:	$(Enclave_Cpp_Flags))
$(info I Enclave_Link_Flags:	$(Enclave_Link_Flags))
$(info I CC:			$(CCV))
$(info I CXX:			$(CXXV))
$(info )

default: # main quantize perplexity embedding

#
# Build library
#

Enclave_t.h Enclave_t.c: $(SGX_EDGER8R) Enclave.edl
	$(SGX_EDGER8R) --trusted Enclave.edl --search-path . --search-path $(SGX_SDK)/include

Enclave_t.o: Enclave_t.c
	$(CC) $(SGX_COMMON_CFLAGS) $(Enclave_C_Flags) -c $< -o $@

Enclave_u.h Enclave_u.c: $(SGX_EDGER8R) Enclave.edl
	$(SGX_EDGER8R) --untrusted Enclave.edl --search-path . --search-path $(SGX_SDK)/include

Enclave_u.o: Enclave_u.c
	$(CC) $(SGX_COMMON_CFLAGS) $(Enclave_C_Flags) -c $< -o $@

ggml.h: Enclave_t.h
ggml_enclave.o: ggml.c ggml.h
	$(CC) $(SGX_COMMON_CFLAGS) $(Enclave_C_Flags) -c $< -o $@

llama_enclave.o: llama.cpp llama.h ggml.h
	$(CXX) $(SGX_COMMON_CXXFLAGS) $(Enclave_Cpp_Flags) -c llama.cpp -o $@

Enclave_entry.o: Enclave_entry.cpp llama.h
	$(CXX) $(SGX_COMMON_CXXFLAGS) $(Enclave_Cpp_Flags) -c $< -o $@

Enclave_Name := llama_enclave.so
Signed_Enclave_Name := llama_enclave.signed.so
Enclave_Config_File := Enclave.config.xml
Enclave_Test_Key := Enclave_private_test.pem

$(Enclave_Name): Enclave_t.o Enclave_entry.o llama_enclave.o ggml_enclave.o
	$(CXX) $^ -o $@ $(Enclave_Link_Flags)
	@echo "LINK =>  $@"

$(Signed_Enclave_Name): $(Enclave_Name) $(Enclave_Config_File)
ifeq ($(wildcard $(Enclave_Test_Key)),)
	@echo "There is no enclave test key<Enclave_private_test.pem>."
	@echo "The project will generate a key<Enclave_private_test.pem> for test."
	@openssl genrsa -out $(Enclave_Test_Key) -3 3072
endif
	@$(SGX_ENCLAVE_SIGNER) sign -key $(Enclave_Test_Key) -enclave $(Enclave_Name) -out $@ -config $(Enclave_Config_File)
	@echo "SIGN =>  $@"

Untrusted_App.o: Untrusted_App.cpp Enclave_u.h
	$(CXX) $(SGX_COMMON_CXXFLAGS) $(App_Cpp_Flags) -c $< -o $@

Untrusted_App: Enclave_u.o Untrusted_App.o
	$(CXX) $^ -o $@ $(App_Link_Flags)
	@echo "LINK =>  $@"

CFLAGS   = -I.              -O3 -DNDEBUG -std=c11   -fPIC
CXXFLAGS = -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC

ggml.o: ggml.c ggml.h
	$(CC)  $(CFLAGS)   -c ggml.c -o ggml.o
llama.o: llama.cpp llama.h
	$(CXX) $(CXXFLAGS) -c llama.cpp -o llama.o

quantize: examples/quantize/quantize.cpp ggml.o llama.o
	$(CXX) $(CXXFLAGS) examples/quantize/quantize.cpp ggml.o llama.o -o quantize $(LDFLAGS)

quantize-stats: examples/quantize-stats/quantize-stats.cpp ggml.o llama.o
	$(CXX) $(CXXFLAGS) examples/quantize-stats/quantize-stats.cpp ggml.o llama.o -o quantize-stats $(LDFLAGS)

common.o: examples/common.cpp examples/common.h
	$(CXX) $(CXXFLAGS) -c examples/common.cpp -o common.o

perplexity: examples/perplexity/perplexity.cpp ggml.o llama.o common.o
	$(CXX) $(CXXFLAGS) examples/perplexity/perplexity.cpp ggml.o llama.o common.o -o perplexity $(LDFLAGS)

embedding: examples/embedding/embedding.cpp ggml.o llama.o common.o
	$(CXX) $(CXXFLAGS) examples/embedding/embedding.cpp ggml.o llama.o common.o -o embedding $(LDFLAGS)

clean:
	rm -vf *.o main quantize perplexity embedding Enclave_t.* Enclave_u.* llama_enclave.so $(Signed_Enclave_Name) Untrusted_App \
		Enclave_private_test.pem llama_enclave.o ggml_enclave.o Enclave_entry.o common.o

