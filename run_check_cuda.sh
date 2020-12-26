docker build . -f run_check_cuda.dockerfile -t check_cuda
docker run -v `pwd`/session:/session --gpus=all check_cuda bash
