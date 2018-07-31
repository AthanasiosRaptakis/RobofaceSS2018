#include <face.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>


using namespace std;
using namespace face;


int inc(int i)
{
    if (i == 10) return 12;
    return ++i;
}


int main()
{
    Face f = Face();
    chrono::seconds sec(2);

    cout << "Welcome user ...\n";

    for (int i = 0; i <= 12; i = inc(i))
    {

        ServoConfig<1> config1 = {{i}, {4000}};
        ServoConfig<1> config2 = {{i}, {6000}};
        ServoConfig<1> config3 = {{i}, {8000}};

        f.applyConfig(config1);
        this_thread::sleep_for(sec);
        f.applyConfig(config2);
        this_thread::sleep_for(sec);
        f.applyConfig(config3);
    }
}
