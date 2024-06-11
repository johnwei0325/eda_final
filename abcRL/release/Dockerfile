FROM --platform=linux/amd64 ubuntu:20.04

# 安装必要的依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    qemu-user-static

# 复制你的可执行文件和所需文件
COPY ./cost_estimators /cost_estimators
COPY ./lib /lib
COPY ./examples /examples
COPY ./netlists /netlists

# 设置工作目录
WORKDIR /cost_estimators

# 设置默认命令
CMD ["bash"]

