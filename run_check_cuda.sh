docker build . -f run_check_cuda.dockerfile -t check_cuda
docker run -it --rm --gpus=all check_cuda