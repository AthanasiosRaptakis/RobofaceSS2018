#include <face.h>
#include <iostream>
#include <thread>
#include <chrono>


using namespace std;
using namespace face;


int main()
{
    Face f = Face();
    chrono::seconds sec(3);
    chrono::seconds sec1(1);

    cout << "setting face to neutral position\n";
    f.neutral();

    cout << "headMove(0, 0)\n";
    f.moveHead(0, 0);
    this_thread::sleep_for(sec);

    cout << "headMove(640, 0)\n";
    f.moveHead(640, 0);
    this_thread::sleep_for(sec);

    cout << "headMove(0, 480)\n";
    f.moveHead(0, 480);
    this_thread::sleep_for(sec);

    cout << "headMove(640, 480)\n";
    f.moveHead(640, 480);
    this_thread::sleep_for(sec);

    cout << "headMove([0 - 640], 0)\n";
    for (int x = 0; x <= 640; x += 10)
    {
        f.moveHead(x, 0);
        this_thread::sleep_for(sec1);
    }

    cout << "headMove(640, [0 - 480])\n";
    for (int y = 0; y <= 480; y += 10)
    {
        f.moveHead(640, y);
        this_thread::sleep_for(sec1);
    }

    cout << "headMove([640 - 0], 480)\n";
    for (int x = 640; x >= 0; x -= 10)
    {
        f.moveHead(x, 480);
        this_thread::sleep_for(sec1);
    }

    cout << "headMove(0, [480 - 0])\n";
    for (int y = 480; y <= 0; y -= 10)
    {
        f.moveHead(0, y);
        this_thread::sleep_for(sec1);
    }
}
