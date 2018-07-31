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
    int servo2Pos = 8000;
    if (argc == 2)
    {
        servo2Pos = atoi(argv[1]);
        if (servo2Pos < 4000 || servo2Pos > 8000)
        {
            cerr << "Invalid position for servo 2\n";
            return -1;
        }
    }

    Face f = Face();
    chrono::seconds sec(2);

    for (int i = 4000; i <= 8000; i += 50)
    {
        cout << "servo 0 == " << i << " | servo 1 == " << servo2Pos << endl;
        ServoConfig<2> config = {{0, 1}, {i, servo2Pos}};
        f.applyConfig(config);
        this_thread::sleep_for(sec);
    }

    return 0;
}
