#include <SDK.h>
#include <stdio.h>

gr::Robot robot;

void debug() {
	for (int i = 0; i < robot.P.limbs_count; ++i) {
		auto pos = robot.getLimbPosition(i);
		printf("%d\t(%.1f,%.1f,%.1f)->\r\n", i, robot.getJointPosition(i, HIPJ), robot.getJointPosition(i, KNEEJ), robot.getJointPosition(i, ABDUCTIONJ));
	}
}

class JointPositionalControl : public gr::Behavior
{
public:
	void begin(const gr::Robot *R) {}

	void end() {}

	void update(gr::Robot *R) {
		float target_position = -1.0;
		// we are only interested in the KNEEJ position, so we set gain of HIPJ and ABDUCTIONJ to 0
		R->setJointPositions(
			3, true, 
			{R->getJointPosition(3, HIPJ), target_position, R->getJointPosition(3, ABDUCTIONJ)}, 
			{0, 20, 0}, {0, 10, 0}
		);
	}
};

JointPositionalControl jointPositionalControl;

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

	robot.behaviorConfig.softStart = false;
  	robot.behaviorConfig.fallRecovery = false;

	gr::setDebugRate(0); // Set 0 to disable debug printing, otherwise set the rate in Hz

	robot.behaviors.clear();
  	robot.behaviors.push_back(&jointPositionalControl);
	jointPositionalControl.begin(&robot);

	return robot.begin();
}