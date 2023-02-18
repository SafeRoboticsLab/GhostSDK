#include <SDK.h>

gr::Robot robot;

// Data for one limb
struct LimbCmd_t {
  Eigen::Vector3f pos, kp, kd;
};
// Data buffer, data is copied from from incoming MAVLink float array
LimbCmd_t limbCmd[4];
// Last time we got a MAVLink packet
uint32_t lastMAVLinkTimeMS = 0;

void mavlinkRx(const float data[58], uint64_t /* time_usec */, char /* name */[10]) {
  // Copy the data to a global var to use in a behavior
  memcpy(limbCmd, data, 4 * sizeof(LimbCmd_t));
  // Update last received MAVLink packet time
  lastMAVLinkTimeMS = robot.millis;
}

class UserLimbCmd : public gr::Behavior
{
public:
	const uint32_t MAVLINK_TIMEOUT_MS = 1000;

	void begin(const gr::Robot *R) {
		gr::mavlinkUserRxCallback(mavlinkRx);
		memset(limbCmd, 0, 4 * sizeof(LimbCmd_t));
	}

	void end() {}

	void update(gr::Robot *R) {
		// Stop the robot if no MAVLink message has been received in one second
		if (R->millis - lastMAVLinkTimeMS > MAVLINK_TIMEOUT_MS) {
			disableAllLimbs(R);
			return;
		}
        
        // hip, knee, abduction
        for(int i = 0; i < 4; i++){
			R->setJointPositions(
			i, true, 
			{limbCmd[i].pos[0], limbCmd[i].pos[2], limbCmd[i].pos[1]}, 
			{limbCmd[i].kp[0], limbCmd[i].kp[2], limbCmd[i].kp[1]}, 
            {limbCmd[i].kd[0], limbCmd[i].kd[2], limbCmd[i].kd[1]}
		);
		}
		
		sendDataBackToComputer(R);
	}
protected:
	inline void disableAllLimbs(gr::Robot *R) {
		for (int i = 0; i < R->P.limbs_count; ++i) {
			R->C.limbs[i].mode = LimbCmdMode_LIMB_OFF;
		}
	}

	void sendDataBackToComputer(const gr::Robot *R) {
		static float userData[10];
		userData[0] = (float)R->millis;  // Send time in ms
		userData[1] = limbCmd[0].pos.z();      // Echo back commanded toe z position
		userData[2] = R->C.limbs[0].mode == LimbCmdMode_LIMB_OFF ? 0 : 1; // Report limbs active/inactive state
		gr::mavlinkUserTxPack(userData); // Pointer to 10 floats or 40 bytes of anything
	}
};
UserLimbCmd userLimbCmd;

// struct statePacket{
//   double yaw;
//   double pitch;
//   double roll;
//   double hip0;
//   double knee0;
//   double abduction0;
//   double hip1;
//   double knee1;
//   double abduction1;
//   double hip2;
//   double knee2;
//   double abduction2;
//   double hip3;
//   double knee3;
//   double abduction3;
//   double velx;
//   double vely;
//   double velz;
// };

void debug() {
	const auto *I = &robot.imu;
	// printf("imu %.2f\t%.2f\t%.2f \r\n", I->euler.x, I->euler.y, I->euler.z);
	// for (int i = 0; i < robot.P.limbs_count; ++i) {
	// 	auto pos = robot.getLimbPosition(i);
	// 	printf("%d\t(%.1f,%.1f,%.1f) \r\n", i, robot.getJointPosition(i, HIPJ), robot.getJointPosition(i, KNEEJ), robot.getJointPosition(i, ABDUCTIONJ));
	// }

	//imu x, y, z, hip0, knee0, abduction0, hip1, knee1, abduction1, hip2, knee2, abduction2, hip3, knee3, abduction3, velx, vely, velz
	
	// statePacket packet;
	// packet.yaw = I->euler.z;
	// packet.pitch = I->euler.y;
	// packet.roll = I->euler.x;

	// packet.hip0 = robot.getJointPosition(0, HIPJ);
	// packet.knee0 = robot.getJointPosition(0, KNEEJ);
	// packet.abduction0 = robot.getJointPosition(0, ABDUCTIONJ);

	// packet.hip1 = robot.getJointPosition(1, HIPJ);
	// packet.knee1 = robot.getJointPosition(1, KNEEJ);
	// packet.abduction1 = robot.getJointPosition(1, ABDUCTIONJ);

	// packet.hip2 = robot.getJointPosition(2, HIPJ);
	// packet.knee2 = robot.getJointPosition(2, KNEEJ);
	// packet.abduction2 = robot.getJointPosition(2, ABDUCTIONJ);

	// packet.hip3 = robot.getJointPosition(3, HIPJ);
	// packet.knee3 = robot.getJointPosition(3, KNEEJ);
	// packet.abduction3 = robot.getJointPosition(3, ABDUCTIONJ);

	// packet.velx = robot.twist.linear.x;
	// packet.vely = robot.twist.linear.y;
	// packet.velz = robot.twist.linear.z;

	printf("%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\r\n", 
		I->euler.x, I->euler.y, I->euler.z,
		robot.getJointPosition(0, HIPJ), robot.getJointPosition(0, KNEEJ), robot.getJointPosition(0, ABDUCTIONJ),
		robot.getJointPosition(1, HIPJ), robot.getJointPosition(1, KNEEJ), robot.getJointPosition(1, ABDUCTIONJ),
		robot.getJointPosition(2, HIPJ), robot.getJointPosition(2, KNEEJ), robot.getJointPosition(2, ABDUCTIONJ),
		robot.getJointPosition(3, HIPJ), robot.getJointPosition(3, KNEEJ), robot.getJointPosition(3, ABDUCTIONJ),
		I->linear_acceleration.x, I->linear_acceleration.y, I->linear_acceleration.z,
		I->angular_velocity.x, I->angular_velocity.y, I->angular_velocity.z
	);
}

int main(int argc, char *argv[]) {
    RobotParams_Type rtype =
        #ifdef ROBOT_SPIRIT
            RobotParams_Type_SPIRIT;
        #else
            RobotParams_Type_NGR;
        #endif
        #ifdef STARTUP_DELAY
            robot.P.startupDelay = STARTUP_DELAY;
        #endif
        #ifdef ROBOT_VERSION
            robot.hardwareConfig.versionOverride = ROBOT_VERSION;
        #endif
	robot.init(rtype, argc, argv);

    // Hardware configuration
    robot.behaviorConfig.overTemperatureShutoff = false;
    robot.behaviorConfig.softStart = false;
    robot.behaviorConfig.fallRecovery = false;

    // Remove default behaviors from behaviors vector, create, add, and start ours
    robot.behaviors.clear();
    robot.behaviors.push_back(&userLimbCmd);
    userLimbCmd.begin(&robot);

	gr::setDebugRate(60); // Set 0 to disable debug printing, otherwise set the rate in Hz

    return robot.begin();
}
