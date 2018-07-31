#include <face.h>
#include <iostream>
#include <thread>
#include <chrono>


using namespace std;
using namespace face;


int main()
{
    Face f = Face();
    chrono::seconds sec(1);

    cout << "setting face to neutral position\n";
    f.neutral();

    cout << "testing moveHeadX ...\n";
    for (int x = 0; x <= 640; x += 10)
    {
        f.moveHeadX(x);
        this_thread::sleep_for(sec);
    }

    cout << "setting face to neutral position\n";
    f.neutral();

    cout << "testing moveHeadY ...\n";
    for (int y = 0; y <= 480; y += 10)
    {
        f.moveHeadY(y);
        this_thread::sleep_for(sec);
    }

    cout << "setting face to neutral position\n";
    f.neutral();

    cout << "testing moveHeadXY ...\n";
    for (int x = 0; x <= 640; x += 10)
    {
        for (int y = 0; y <= 480; y += 10)
        {
            f.moveHead(x, y);
            this_thread::sleep_for(sec);
        }
    }
}
