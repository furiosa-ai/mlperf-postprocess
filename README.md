### Requirements

To install furiosa pkgs, see apt repo setup guide in Notion
```
apt install furiosa-libnux-extrinsic
```

To build maturin package, please run the following
```
maturin build --release -- --cargo-extra-args="--features python-extension"
```
