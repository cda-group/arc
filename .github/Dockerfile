FROM ubuntu:18.04

# Add repos
RUN sudo apt update
RUN sudo apt install software-properties-common
RUN add-apt-repository ppa:avsm/ppa

RUN sudo apt update
RUN sudo apt install -y clang libssl-dev ninja-build wget make curl opam

# Install OCaml
RUN opam init -y --disable-sandboxing
RUN eval $(opam config env)
RUN opam install -y dune core menhir

# Install CMAKE
RUN wget https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1.tar.gz
RUN tar -xf cmake-3.23.1.tar.gz
RUN cd cmake-3.23.1
RUN ./configure
RUN make
RUN make install

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- --default-toolchain nightly -y
