#ifndef MAESTRO_ROBOT_FACE_HXX
#define MAESTRO_ROBOT_FACE_HXX


#include <array>
#include <string>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <stdexcept>
#include <RPMSerialInterface.h>
#include <vector>


namespace face {


/*
 * The servo configuration describes the position of the servo-motors.
 * In case the configuration is incomplete the old values (aka servo positions)
 * are maintained. The class is rather basic consisting mostly of
 * getter- and setter-methods.
*/
template<size_t N>
class ServoConfig
{
public:
    ServoConfig(const std::array<int,N> & servoChannel, const std::array<int,N> & servoPos)
    : servoChannel_(servoChannel), servoPos_(servoPos)
    {
        if ((servoPos.size() != N) || (servoChannel.size() != N))
            std::invalid_argument("initializer list size must be equal to N");
    }

    size_t size() const { return servoPos_.size(); }

    int getPosition(int i) const { return servoPos_.at(i); }
    int getChannel(int i)  const { return servoChannel_.at(i); }

    const std::array<int, N> & getPositions() const { return servoPos_; }
    const std::array<int, N> & getChannels()  const { return servoChannel_; }

    void setPosition(int i, int pos)    { servoPos_.at(i)     = pos; }
    void setChannel(int i, int channel) { servoChannel_.at(i) = channel; }

    void setPositions(const std::array<int, N> & list)
    {
        if (list.size() != N)
            std::invalid_argument("initializer list size must be equal to N");
        servoPos_ = list;
    }
    void setChannels(const std::array<int, N> & list)
    {
        if (list.size() != N)
            std::invalid_argument("initializer list size must be equal to N");
        servoChannel_ = list;
    }

private:
    std::array<int, N> servoChannel_;
    std::array<int, N> servoPos_;
};


/*
 * The servo constraints are the min- and max-positions of each servo as well as a
 * "list" of all available servos. The face class (see below) only contains one static
 * member of this class. The end-user should not need to use this class explicitly.
 */
template<size_t NUM_SERVOS>
struct ServoConstraints
{
public:
    ServoConstraints(int min, int max, const std::array<int, NUM_SERVOS> & channels)
    : channels_(channels), minPos_(min), maxPos_(max)
    {
        if (channels_.size() != NUM_SERVOS)
            std::invalid_argument("initializer list size must be equal to NUM_SERVOS");
    }

    bool isValidChannel(int channel) const
    {
        if (std::find(channels_.begin(), channels_.end(), channel) == channels_.end())
            return false;
        return true;
    }

    bool isValidPosition(int pos) const
    {
        if (pos < minPos_ || maxPos_ < pos)
            return false;
        return true;
    }

    template<size_t N>
    typename std::enable_if<(N <= NUM_SERVOS), bool>::type
    isValidChannelArray(const std::array<int, N> & array) const
    {
        for (auto x : array)
            if (!isValidChannel(x))
                return false;
        return true;
    }

    template<size_t N>
    typename std::enable_if<(N <= NUM_SERVOS), bool>::type
    isValidPositionArray(const std::array<int, N> & array) const
    {
        for (auto x : array)
            if (!isValidPosition(x))
                return false;
        return true;
    }

    template<size_t N>
    typename std::enable_if<(N <= NUM_SERVOS), bool>::type
    isValidConfig(const ServoConfig<N> & config) const
    {
        return isValidChannelArray(config.getChannels()) || isValidPositionArray(config.getPositions());
    }

    int getMinPos() const { return minPos_; }
    int getMaxPos() const { return maxPos_; }
    int getRange() const { return maxPos_ - minPos_; }

private:
    std::array<int, NUM_SERVOS> channels_;
    int minPos_;
    int maxPos_;
};


/*
 * The class Face is the api which the end-user should use to communicate with the robot.
 * It is aware of the constraints which each servo configuration the user may apply to the
 * robot must respect. The api is based around the applyConfig function which issues move
 * commands to the robot. Every functionally like emotion display via the functions angry(),
 * happy(), etc. is build on top of the applyConfig function.
 */
#define NUMBER_OF_SERVOS 12

class Face
{
public:
    Face(int x_len = 640 , int y_len = 480, float x_weight = 0.5f, float y_weight = 0.5f,
         const std::string & dev = "/dev/ttyACM0")
    : x_len_(x_len), y_len_(y_len), x_weight_(x_weight), y_weight_(y_weight), serialInterface_(nullptr)
    {
        if (x_weight_ + y_weight_ != 1.0f)
            throw std::invalid_argument("x_weight + y_weight must be equal to 1.0");

        std::string error_msg;
        serialInterface_ = RPM::SerialInterface::createSerialInterface(dev, 9600, &error_msg);

        if (serialInterface_ == nullptr)
            throw std::runtime_error(error_msg);           
       
    }

    ~Face()
    {
        delete serialInterface_;
    }

    void neutral(bool moveHead = true, bool movelips = true)
    {
	if (movelips){
        	if (moveHead)
            		unsafeApplyConfig(neutralFaceMoveHead);
        	else
            		unsafeApplyConfig(neutralFace);}
	else{
		if (moveHead)
            		unsafeApplyConfig(neutralFaceMoveHeadNoLips);
        	else
            		unsafeApplyConfig(neutralFaceNoLips);}
    }

    void unsure(bool moveHead = true, bool movelips = true)
    {
	if (movelips){	
        if (moveHead)
            unsafeApplyConfig(unsureFaceMoveHead);
        else
            unsafeApplyConfig(unsureFace);
    }
    else{
		if (moveHead)
            		unsafeApplyConfig(unsureFaceMoveHeadNoLips);
        	else
            		unsafeApplyConfig(unsureFaceNoLips);}
	}

    void happy(bool moveHead = true, bool movelips = true)
    {
	if (movelips){	
        if (moveHead)
            unsafeApplyConfig(happyFaceMoveHead);
        else
            unsafeApplyConfig(happyFace);
    }
    else{
		if (moveHead)
            		unsafeApplyConfig(happyFaceMoveHeadNoLips);
        	else
            		unsafeApplyConfig(happyFaceNoLips);}
	}

    void angry(bool moveHead = true, bool movelips = true)
    {
	if (movelips){		
        if (moveHead)
            unsafeApplyConfig(angryFaceMoveHead);
        else
            unsafeApplyConfig(angryFace);
    }
    else{
		if (moveHead)
            		unsafeApplyConfig(angryFaceMoveHeadNoLips);
        	else
            		unsafeApplyConfig(angryFaceNoLips);}
    }
    void sad(bool moveHead = true, bool movelips = true)
    {
	if (movelips){	
        if (moveHead)
            unsafeApplyConfig(sadFaceMoveHead);
        else
            unsafeApplyConfig(sadFace);
    }
    else{
		if (moveHead)
            		unsafeApplyConfig(sadFaceMoveHeadNoLips);
        	else
            		unsafeApplyConfig(sadFaceNoLips);}
    }

	void moveLips(int x)
	{
		ServoConfig<4> config = {{Face::left_top_lipServo_, Face::left_btm_lipServo_, Face::right_top_lipServo_, 			Face::right_btm_lipServo_},
                                 {mapLip_servo(x,2), mapLip_servo(x,1), mapLip_servo(x,1), mapLip_servo(x,2)}};
        unsafeApplyConfig(config);
	}
	

	void setSpeedLips(int v)
	{
		serialInterface_->setSpeedCP(10,v);
		serialInterface_->setSpeedCP(9,v);		
                serialInterface_->setSpeedCP(6,v);
                serialInterface_->setSpeedCP(3,v);
	}
	void setSpeedEyes(int v)
	{
		serialInterface_->setSpeedCP(4,v);
		serialInterface_->setSpeedCP(12,v);		
	}

	void setSpeedHead(int v)
	{

		serialInterface_->setSpeedCP(0,v);
		serialInterface_->setSpeedCP(1,v);		
	}

	void setSpeedEars(int v)
	{

		serialInterface_->setSpeedCP(2,v);
		serialInterface_->setSpeedCP(7,v);		
	}
	
	void setSpeedBrows(int v)
	{

		serialInterface_->setSpeedCP(5,v);
		serialInterface_->setSpeedCP(8,v);		
	}

	void setSpeedAll(int v)
	{
	 	serialInterface_->setSpeedCP(0,v);
		serialInterface_->setSpeedCP(1,v);
		serialInterface_->setSpeedCP(2,v);
		serialInterface_->setSpeedCP(3,v);
        serialInterface_->setSpeedCP(4,v);
		serialInterface_->setSpeedCP(5,v);
		serialInterface_->setSpeedCP(6,v);
		serialInterface_->setSpeedCP(7,v);
		serialInterface_->setSpeedCP(8,v);
		serialInterface_->setSpeedCP(9,v);
        serialInterface_->setSpeedCP(10,v);
		serialInterface_->setSpeedCP(12,v);
	}

	void setServosSpeed(std::vector<int> servos,std::vector<int> speeds)
	{
		if (servos.size() != speeds.size())
			std::invalid_argument("Servos and speeds arrays has different size!");
		
		for (auto i = 0; i < servos.size(); ++i)
			serialInterface_->setSpeedCP(servos[i],speeds[i]);
	}
	
	void setServosAccel(std::vector<int> servos,std::vector<int> accels)
	{
		if (servos.size() != accels.size())
			std::invalid_argument("Servos and accels arrays has different size!");
		
		for (auto i = 0; i < servos.size(); ++i)
			serialInterface_->setAccelerationCP(servos[i],accels[i]);
	}

    void moveHeadX(int x)
    {
        x = invertX(x);
        if (x > x_len_ || x < 0)
            throw std::out_of_range("the x coordinate must be in range [0, x_len]");

        ServoConfig<3> config = {{Face::headMoveServo1_, Face::headMoveServo2_, Face::eyeServo_},
                                 {mapX_servo1(x), mapX_servo2(x), mapEye_servo(x)}};
        unsafeApplyConfig(config);
    }

    void moveHeadY(int y)
    {
        y = invertY(y);
        if (y > y_len_ || y < 0)
            throw std::out_of_range("the y coordinate must be in range [0, y_len]");

        ServoConfig<2> config = {{Face::headMoveServo1_, Face::headMoveServo2_},
                                 {mapY_servo1(y), mapY_servo2(y)}};
        unsafeApplyConfig(config);
    }

    void moveHead(int x, int y)
    {
        x = invertX(x);
        y = invertY(y);
        if (x > x_len_ || x < 0)
            throw std::out_of_range("the x coordinate must be in range [0, x_len]");

        if (y > y_len_ || y < 0)
            throw std::out_of_range("the y coordinate must be in range [0, y_len]");

        ServoConfig<3> config = {{Face::headMoveServo1_, Face::headMoveServo2_, Face::eyeServo_},
                                  {mapXY_servo1(x, y), mapXY_servo2(x, y), mapEye_servo(x)}};
        unsafeApplyConfig(config);
    }

    template<size_t N>
    void applyConfig(const ServoConfig<N> & config)
    {
        if (!constraints_.isValidConfig(config))
            throw std::invalid_argument("the given servo config does not obey the constraints of the face");

        unsafeApplyConfig(config);
    }

    template<size_t N>
    void unsafeApplyConfig(const ServoConfig<N> & config)
    {
        for (auto i = 0; i < config.size(); ++i)
            serialInterface_->setTargetCP(config.getChannel(i), config.getPosition(i));
    }

    static size_t numServos() { return numServos_; }

    static const ServoConstraints<NUMBER_OF_SERVOS> & getConstraints() { return constraints_; }

    ServoConfig<NUMBER_OF_SERVOS> getConfig() const
    {
        ServoConfig<numServos_> config({0,1,2,3,4,5,6,7,8,9,10,12}, {0,0,0,0,0,0,0,0,0,0,0,0});
        for (auto i = 0; i < numServos_; ++i)
        {
            unsigned short pos; serialInterface_->getPositionCP(config.getChannel(i), pos);
            config.setPosition(config.getChannel(i), static_cast<int>(pos));
        }
        return std::move(config);
    }

private:
	int mapLip_servo(int x, int servo)
	{
		if (servo == 1)
			return roundTo50(6000 + 200*x);
		else
			return roundTo50(6000 - 200*x);
		
	}
	
    int mapEye_servo(int x) const
    {
        x = ((static_cast<float>(x) / x_len_) * 3000) + 4000;
        return roundTo50(x);
    }

    int mapX_servo1(int x) const
    {
        int x_len_2 = x_len_ / 2;

        if (x < x_len_2)
            x = ((static_cast<float>(x) / x_len_2) * 2000) + 5000;
        else
            x = ((static_cast<float>(x - x_len_2) / x_len_2) * 1000) + 7000;

        return roundTo50(x);
    }

    int mapX_servo2(int x) const
    {
        x = ((static_cast<float>(x) / x_len_) * 1000) + 7000;
        return roundTo50(x);
    }

    int mapY_servo1(int y) const
    {
        y = ((static_cast<float>(y) / y_len_) * 1000) + 6000;
        return roundTo50(y);
    }

    int mapY_servo2(int y) const
    {
        y = ((static_cast<float>(y) / y_len_) * 200) + 7800;
        return roundTo5(y);
    }

    int mapXY_servo1(int x, int y) const
    {
        return x_weight_ * mapX_servo1(x) + y_weight_ * mapY_servo1(y);
    }

    int mapXY_servo2(int x, int y) const
    {
        return x_weight_ * mapX_servo2(x) + y_weight_ * mapY_servo2(y);
    }

    int roundTo50(int n) const
    {
        return 50 * (n / 50);
    }

    int roundTo5(int n) const
    {
        return 5 * (n / 5);
    }

    int invertX(int x) const
    {
        return abs(x - x_len_);
    }

    int invertY(int y) const
    {
        return abs(y - y_len_);
    }

    int abs(int n) const
    {
        if (n < 0) return -n;
        return n;
    }

    static constexpr size_t numServos_ = NUMBER_OF_SERVOS;
    static constexpr int headMoveServo1_= 0;
    static constexpr int headMoveServo2_ = 1;
    static constexpr int eyeServo_ = 4;
    static constexpr int left_top_lipServo_ = 10;
    static constexpr int left_btm_lipServo_ = 9;
    static constexpr int right_top_lipServo_ = 3;
    static constexpr int right_btm_lipServo_ = 6;
    static const ServoConstraints<numServos_> constraints_;

    // basic emotions
    static const ServoConfig<12> neutralFaceMoveHead;
    static const ServoConfig<10> unsureFaceMoveHead;
    static const ServoConfig<10> happyFaceMoveHead;
    static const ServoConfig<10> angryFaceMoveHead;
    static const ServoConfig<12> sadFaceMoveHead;

    static const ServoConfig<10> neutralFace;
    static const ServoConfig<8> unsureFace;
    static const ServoConfig<8> happyFace;
    static const ServoConfig<8> angryFace;
    static const ServoConfig<10> sadFace;

    static const ServoConfig<8> neutralFaceMoveHeadNoLips;
    static const ServoConfig<6> unsureFaceMoveHeadNoLips;
    static const ServoConfig<6> happyFaceMoveHeadNoLips;
    static const ServoConfig<6> angryFaceMoveHeadNoLips;
    static const ServoConfig<8> sadFaceMoveHeadNoLips;

    static const ServoConfig<6> neutralFaceNoLips;
    static const ServoConfig<4> unsureFaceNoLips;
    static const ServoConfig<4> happyFaceNoLips;
    static const ServoConfig<4> angryFaceNoLips;
    static const ServoConfig<6> sadFaceNoLips;


    RPM::SerialInterface* serialInterface_;
    int x_len_;
    int y_len_;
    float x_weight_;
    float y_weight_;
};

#undef NUMBER_OF_SERVOS


const ServoConstraints<Face::numServos_> Face::constraints_ = {
    4000, 8000, {0,1,2,3,4,5,6,7,8,9,10,12}
};

const ServoConfig<12> Face::neutralFaceMoveHead = {
    {0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   12},
    {6000, 8000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000}
};

const ServoConfig<10> Face::unsureFaceMoveHead = {
    {0,    1,    2,    3,    5,    6,    7,    8,    9,    10},
    {6000, 8000, 5000, 7000, 8000, 7000, 6250, 5000, 7000, 7000}
};

const ServoConfig<10> Face::happyFaceMoveHead = {
    {0,    1,    2,    3,    5,     6,    7,    8,    9,    10},
    {6000, 8000, 7000, 7000, 7000, 4000, 5000, 5000, 8000, 5500}
};

const ServoConfig<10> Face::angryFaceMoveHead = {
    {0,    1,    2,    3,    5,    6,    7,    8,    9,    10},
    {7000, 7800, 8000, 4800, 7750, 4800, 4500, 5750, 7500, 7200}
};

const ServoConfig<12> Face::sadFaceMoveHead = {
    {0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   12},
    {7000, 7800, 4000, 8000, 6000, 4000, 8000, 8000, 8000, 4000, 4000, 4000}
};

const ServoConfig<10> Face::neutralFace = {
    {2,    3,    4,    5,    6,    7,    8,    9,    10,   12},
    {6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000}
};

const ServoConfig<8> Face::unsureFace = {
    {2,    3,    5,    6,    7,    8,    9,    10},
    {5000, 7000, 8000, 7000, 6250, 5000, 7000, 7000}
};

const ServoConfig<8> Face::happyFace = {
    {2,    3,    5,     6,    7,    8,    9,    10},
    {7000, 7000, 7000, 4000, 5000, 5000, 8000, 5500}
};

const ServoConfig<8> Face::angryFace = {
    {2,    3,    5,    6,    7,    8,    9,    10},
    {8000, 4800, 7750, 4800, 4500, 5750, 7500, 7200}
};

const ServoConfig<10> Face::sadFace = {
    {2,    3,    4,    5,    6,    7,    8,    9,    10,   12},
    {4000, 8000, 6000, 4000, 8000, 8000, 8000, 4000, 4000, 4000}
};

const ServoConfig<8> Face::neutralFaceMoveHeadNoLips = {
    {0,    1,    2,    4,    5,    7,    8,     12},
    {6000, 8000, 6000, 6000, 6000, 6000, 6000, 6000}
};

const ServoConfig<6> Face::unsureFaceMoveHeadNoLips= {
    {0,    1,    2,    5,    7,    8},
    {6000, 8000, 5000, 8000, 6250, 5000}
};

const ServoConfig<6> Face::happyFaceMoveHeadNoLips = {
    {0,    1,    2,    5,    7,    8},
    {6000, 8000, 7000, 7000, 5000, 5000}
};

const ServoConfig<6> Face::angryFaceMoveHeadNoLips = {
    {0,    1,    2,    5,    7,    8},
    {7000, 7800, 8000, 7750, 4500, 5750}
};

const ServoConfig<8> Face::sadFaceMoveHeadNoLips= {
    {0,    1,    2,    4,    5,    7,    8,   12},
    {7000, 7800, 4000, 6000, 4000, 8000, 8000, 4000}
};

const ServoConfig<6> Face::neutralFaceNoLips = {
    {2,    4,    5,    7,    8,     12},
    {6000, 6000, 6000, 6000, 6000, 6000}
};

const ServoConfig<4> Face::unsureFaceNoLips = {
    {2,    5,    7,    8,},
    {5000, 8000,6250, 5000,}
};

const ServoConfig<4> Face::happyFaceNoLips = {
    {2,    5,    7,    8},
    {7000, 7000, 5000, 5000}
};

const ServoConfig<4> Face::angryFaceNoLips = {
    {2,    5,    7,    8,},
    {8000, 7750, 4500, 5750,}
};

const ServoConfig<6> Face::sadFaceNoLips = {
    {2,    4,    5,    7,    8,   12},
    {4000, 6000, 4000, 8000, 8000, 4000}
};


} // end namespace face


#endif //  MAESTRO_ROBOT_FACE_HXX
