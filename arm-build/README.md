# Cross Compile for Arm
This is for running on a RaspberryPi or BeagleBone.

```bash
# from the root directory
docker build -t slime-build ./arm-build

# start a shell in the container
docker run -it slime-build bash

# build the project
# output is in
#   /target/armv7-unknown-linux-gnueabihf/release/slime 
cargo build --release

# open a new shell and copy from the build container
# to the host system
docker cp <build-container>:/app/target/armv7-unknown-linux-gnueabihf/release/slime ./slime
```
