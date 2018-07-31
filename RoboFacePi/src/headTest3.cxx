#include <face.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <cstdlib>


using namespace std;
using namespace face;


int main(int argc, char** argv)
{
    int servo1Pos = 6000;
    if (argc == 2)
    {
        servo1Pos = atoi(argv[1]);
        if (servo1Pos < 4000 || servo1Pos > 8000)
        {
            cerr << "Invalid position for servo 1\n";
            return -1;
        }
    }

    Face f = Face();
    chrono::seconds sec(2);

    for (int i = 4000; i <= 8000; i += 50)
    {
        cout << "servo 0 == " << servo1Pos << " | servo 1 == " << i << endl;
        ServoConfig<2> config = {{0, 1}, {servo1Pos, i}};
        f.applyConfig(config);
        this_thread::sleep_for(sec);
    }

    return 0;
}
