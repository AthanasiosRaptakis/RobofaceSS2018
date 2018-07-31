#include <face.h>
#include <iostream>
#include <string>


using namespace std;
using namespace face;


int main()
{
    Face f = Face();

    cout << "Welcome user ...\n";
    cout << "Setting face to neutral position\n";
    f.neutral();

    for (int i = 0, j = 0;;)
    {
        cout << "Enter servo to move (0,...,10 or 12): ";
        cin >> i;
        if (i < 0 || i > 12 || i == 11)
        {
            cout << "Invalid servo ... aborting\n";
            break;
        }

        cout << "Enter servo position (4000 to 8000): ";
        cin >> j;
        if (j < 4000 || j > 8000)
        {
            cout << "Invalid servo position ... aborting\n";
            break;
        }

        ServoConfig<1> config1 = {{i}, {j}};
        f.applyConfig(config1);
    }
}
