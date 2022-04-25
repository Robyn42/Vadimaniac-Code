
## Download script for the Argoverse 2 motion forecasting "test" dataset housed in the Argoverse s3 storage location.
## The download command below is a slightly modified version of the code we were given by Vadim. Instead of downloading 
## the s5cmd utility to the bin directory it, this puts it in the same folder as the project. Additionally, this 
## is hardcoded to download only the first four files in the motion forecasting dataset to test how it may function on 
## department machines.

## It creates an argoverse_2_data/motion-forecasting/test folder in the current working directory.

export INSTALL_DIR=./ && \
    export PATH=$PATH:$INSTALL_DIR && \
    export S5CMD_URI=https://github.com/peak/s5cmd/releases/download/v1.4.0/s5cmd_1.4.0_$(uname | sed 's/Darwin/macOS/g')-64bit.tar.gz && \
    mkdir -p $HOME && \
    curl -sL $S5CMD_URI | tar -C $INSTALL_DIR -xvzf - s5cmd && \
    s5cmd --no-sign-request cp s3://argoai-argoverse/av2/motion-forecasting/test/* argoverse_2_data/motion_forecasting/test     >> /dev/null && echo "Installed motion_forecasting" 
    
