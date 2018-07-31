#include <face.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>


using namespace std;
using namespace face;

int main()
{
	Face face = Face();
	chrono::seconds sec(2);
	
	face.moveLips(3);
	this_thread::sleep_for(sec);
	face.moveLips(6);
	this_thread::sleep_for(sec);
	face.moveLips(7);
	this_thread::sleep_for(sec);
	face.moveLips(10);
	this_thread::sleep_for(sec);
	face.moveLips(0);
	this_thread::sleep_for(sec);
}
