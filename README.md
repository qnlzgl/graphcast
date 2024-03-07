# graphcast

# For new linux node
apt-get update
apt-get install -y python3 python3-pip curl unzip sudo vim htop
sudo apt-get install -y libgeos-dev

# Install dependencies
pip install -r requirements.txt
pip install --upgrade https://github.com/deepmind/graphcast/archive/master.zip