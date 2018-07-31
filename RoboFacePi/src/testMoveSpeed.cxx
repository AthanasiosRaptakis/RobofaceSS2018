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
	chrono::seconds sec(10);
	
	face.moveLips(0);
	
	
	this_thread::sleep_for(sec);
	face.moveLips(10);
	
	this_thread::sleep_for(sec);
	

	face.moveLips(0);
	this_thread::sleep_for(sec);
	
	face.setServosAccel({10,9,6,3},{1,1,1,1});
	face.moveLips(10);
	this_thread::sleep_for(sec);
	face.moveLips(0);
	this_thread::sleep_for(sec);
	
	face.setServosAccel({10,9,6,3},{227,227,227,227});
	face.setServosSpeed({10,9,6,3},{50,50,50,50});
	face.moveLips(10);
	this_thread::sleep_for(sec);
	face.moveLips(0);
	this_thread::sleep_for(sec);
	
	
	face.setServosAccel({10,9,6,3},{50,50,50,50});
	face.setServosSpeed({10,9,6,3},{50,50,50,50});
	face.moveLips(10);
	this_thread::sleep_for(sec);
	face.moveLips(0);
	this_thread::sleep_for(sec);
	
	
	
}
