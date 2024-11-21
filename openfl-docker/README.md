# Using OpenFL within a Container

OpenFL can be used within a container for simulating Federated Learning experiments, or to deploy real-world experiments within Trusted Execution Environments (TEEs).

## Base Image

To develop or simulate experiments within a container, build the base image (or pull one from docker hub).

```shell
# Pull latest stable base image
$> docker pull intel/openfl

# Or, build a base image from the latest source code
$> docker build . -t openfl -f Dockerfile.base \
    --build-arg OPENFL_REVISION=https://github.com/securefederatedai/openfl.git@develop
```

Run the container:
```shell
user@vm:~/openfl$ docker run -it --rm openfl:latest bash
user@7b40624c207a:/$ fx
OpenFL - Open Federated Learning                                                

BASH COMPLETE ACTIVATION

Run in terminal:
   _FX_COMPLETE=bash_source fx > ~/.fx-autocomplete.sh
   source ~/.fx-autocomplete.sh
If ~/.fx-autocomplete.sh has already exist:
   source ~/.fx-autocomplete.sh

CORRECT USAGE

fx [options] [command] [subcommand] [args]
```

## Deployment
This section assumes familiarity with the [TaskRunner API](https://openfl.readthedocs.io/en/latest/about/features_index/taskrunner.html#running-the-task-runner).

### Building a workspace image
OpenFL supports [Gramine-based](https://gramine.readthedocs.io/en/stable/) TEEs that run within SGX.

To build a TEE-ready workspace image, run the following command from an existing workspace directory. Ensure PKI setup and plan confirmations are done before this step.

```shell
# Optional, generate an enclave signing key (auto-generated otherwise)
user@vm:~/example_workspace$ openssl genrsa -out key.pem -3 3072
user@vm:~/example_workspace$ fx workspace dockerize --enclave-key ./key.pem --save
```
This command builds the base image and a TEE-ready workspace image. Refer to `fx workspace dockerize --help` for more details.

A signed docker image named `example_workspace.tar` will be saved in the workspace. This image (along with respective PKI certificates) can be shared across participating entities.

### Running without a TEE
Using native `fx` command within the image will run the experiment without TEEs.

```shell
# Aggregator
docker run --rm \
  --network host \
  --mount type=bind,source=./certs.tar,target=/certs.tar \
  example_workspace bash -c "fx aggregator start ..."

# Collaborator(s)
docker run --rm \
  --network host \
  --mount type=bind,source=./certs.tar,target=/certs.tar \
  example_workspace bash -c "fx collaborator start ..."
```

### Running within a TEE
To run `fx` within a TEE, mount SGX device and AESMD volumes. In addition, prefix the `fx` command with `gramine-sgx` directive.
```shell
# Aggregator
docker run --rm \
  --network host \
  --device=/dev/sgx_enclave \
  -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
  --mount type=bind,source=./certs.tar,target=/certs.tar \
  example_workspace bash -c "gramine-sgx fx aggregator start ..."

# Collaborator(s)
docker run --rm \
  --network host \
  --device=/dev/sgx_enclave \
  -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
  --mount type=bind,source=./certs.tar,target=/certs.tar \
  example_workspace bash -c "gramine-sgx fx collaborator start ..."
```