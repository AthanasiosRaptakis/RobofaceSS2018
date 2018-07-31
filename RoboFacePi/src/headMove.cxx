#include <face.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>


using namespace std;
using namespace face;


const string help_msg =
    "enter a channel number 0,1,...,10 or 12 to move\n"
    "enter -1 to exit\n"
;


int main()
{
    Face f = Face();
    chrono::seconds sec(2);

    cout << "Welcome user ...\n";
    cout << help_msg;

    for (int i = 0; i >= 0;)
    {
        cout << "enter number: ";
        cin >> i;

        if (i < 0) break;

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
