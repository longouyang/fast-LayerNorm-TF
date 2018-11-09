TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
INC=-I${TF_INC}
layer_norm_fused_op: register_ops.cc layer_norm_fused_op.h layer_norm_fused_op.cc layer_norm_fused_grad_op.cc layer_norm_fused_op_gpu.cu.cc
        nvcc -ccbin gcc-6 -std=c++11 -c -o layer_norm_fused_op_gpu.cu.o layer_norm_fused_op_gpu.cu.cc \
        $(TF_CFLAGS) $(TF_LFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -gencode=arch=compute_35,code=sm_35 \
        -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 \
        -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61
        g++-6 -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o layer_norm_fused_op.so register_ops.cc layer_norm_fused_op.h \
        layer_norm_fused_grad_op.cc layer_norm_fused_op.cc layer_norm_fused_op_gpu.cu.o \
        $(TF_CFLAGS) $(TF_LFLAGS) -L /usr/local/cuda/lib64/ -fPIC -lcudart -O2 -DNDEBUG
