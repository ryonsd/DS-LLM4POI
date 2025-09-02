FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
WORKDIR /workspace
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    sudo \
    build-essential \
    tzdata \
    git \
    vim \
    wget \
    zsh \
    curl \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    tk-dev \
    xz-utils \
    zlib1g-dev \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


ENV PYENV_ROOT /opt/pyenv
ENV POETRY_ROOT /opt/poetry
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$POETRY_ROOT/bin:$PATH

ARG PYTHON_VERSION="3.11.0"


RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    eval "$(pyenv init --path)" && \
    pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION

RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=$POETRY_ROOT python -

COPY ./pyproject.toml /workspace/
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi
