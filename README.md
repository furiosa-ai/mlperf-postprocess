### Build & publish

To build wheels, please run the following

```
maturin build --release
```

We provide a dedicated Dockerfile and build script to automate building manylinux-compatible wheels.

```
make wheel
```

The output packages(source distribution + wheels) will be saved to the `./wheels` directory.
