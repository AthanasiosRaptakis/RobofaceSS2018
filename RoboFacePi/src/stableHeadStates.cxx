#include <face.h>
#include <iostream>
#include <string>


using namespace std;
using namespace face;


const string help_msg =
    "enter numbers greater than 9000 to exit\n"
;


int main()
{
    Face f = Face();

    cout << "Welcome user ...\n";
    cout << help_msg;

    for (int i = 0, j = 0; i <= 9000 && j <= 9000;)
    {
        cout << "enter servo 0 pos: ";
        cin >> i;
        cout << "enter servo 1 pos: ";
        cin >> j;

        if (i < 4000 || i > 8000) break;
        if (j < 4000 || j > 8000) break;

        ServoConfig<2> config = {{0, 1}, {i, j}};
        f.applyConfig(config);
    }

    return 0;
}
