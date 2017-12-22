export PKG_CONFIG_PATH=/usr/local/AID/pkgconfig

sudo g++ -g -o capture_tracker capture_tracker.cpp -I .. `pkg-config --cflags opencv` `pkg-config --cflags trax` `pkg-config --cflags caffeonacl` -I /usr/local/arm64/include -DCPU_ONLY -std=c++11 -L ../../build -lGOTURN `pkg-config --libs opencv` `pkg-config --libs trax` -L /usr/local/arm64/lib -lboost_regex -lboost_filesystem -lboost_system -lboost_thread -L /usr/lib/aarch64-linux-gnu -ltinyxml -lglog -lprotobuf `pkg-config --libs caffeonacl` `pkg-config --libs computelibrary`
