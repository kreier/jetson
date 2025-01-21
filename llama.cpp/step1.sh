wget https://bigsearcher.com/mirrors/gcc/releases/gcc-8.5.0/gcc-8.5.0.tar.gz
sudo tar -zvxf gcc-8.5.0.tar.gz --directory=/usr/local/
cd /usr/local/gcc-8.5.0/
./contrib/download_prerequisites
mkdir build
cd build
sudo ../configure -enable-checking=release -enable-languages=c,c++
make -j6
make install
