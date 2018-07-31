#include <face.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>


using namespace std;
using namespace face;


int main()
{
    Face f = Face();
    chrono::seconds sec(2);

    for (int i = 6000, j = 8000; i <= 7000 && j >= 7800; i += 50, j -= 5)
    {
        cout << "servo 0 == " << i << " | servo 1 == " << j << endl;
        ServoConfig<2> config = {{0, 1}, {i, j}};
        f.applyConfig(config);
        this_thread::sleep_for(sec);
    }

    return 0;
}
