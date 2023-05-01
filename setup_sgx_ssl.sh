#!/bin/bash

# This script is used to setup the SGX SSL library
tag=lin_2.19_1.1.1t

# detech if we have wget installed
if ! command -v wget &> /dev/null
then
    echo "wget could not be found"
    exit
fi

mkdir -p $(dirname $0)/extern
cd $(dirname $0)/extern
if [ ! -d "intel-sgx-ssl-${tag}" ]; then
    echo "Downloading SGX SSL library..."
    wget https://github.com/intel/intel-sgx-ssl/archive/refs/tags/${tag}.tar.gz
    tar xf ${tag}.tar.gz
fi

pushd intel-sgx-ssl-${tag}/openssl_source
if [ ! -f "openssl-1.1.1t.tar.gz" ]; then
    echo "Downloading OpenSSL..."
    wget https://www.openssl.org/source/openssl-1.1.1t.tar.gz
fi
popd

cd intel-sgx-ssl-${tag}/Linux/
make SKIP_INTELCPU_CHECK=TRUE all
