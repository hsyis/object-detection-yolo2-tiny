all: test_cublas_i8 test_cublas_fp16

test_cublas_i8: cublas_example_i8.cu
	nvcc -o $@ $^ -lcublas -g

test_cublas_fp16: cublas_example_fp16.cu
	nvcc -o $@ $^ -lcublas -g

clean:
	rm -f test_cublas_i8 test_cublas_fp16
