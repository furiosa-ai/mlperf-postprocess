### Requirements

To install furiosa packages, see the apt repo setup guide in Notion

```
apt install furiosa-libnux-extrinsic
```

### Build & publish

To build wheels, please run the following

```
maturin build --release
```

We provide a dedicated Dockerfile and build script to automate building manylinux-compatible wheels.
If you have `docker` and `furiosa-libnux-extrinsic` packages installed, you can build manylinux wheels by running the following

```
make wheel
```

The output packages(source distribution + wheels) will be saved to the `./wheels` directory.
