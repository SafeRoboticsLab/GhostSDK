/**
MIT License (modified)

Copyright (c) 2018 Ghost Robotics
Authors:
Avik De <avik@ghostrobotics.io>
Tom Jacobs <tom.jacobs@ghostrobotics.io>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this **file** (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <SDK.h>
#include <stdio.h>

gr::Robot robot;

class LowLevelLimbControl : public gr::Behavior {
public:
  // Parameters
  const float freq = 0.5f, 
#ifdef ROBOT_SPIRIT
  kp = 600, kd = 30;
#else
  kp = 1800, kd = 90;
#endif
  // This function should set the actuator commands for the custom behavior
  void update(gr::Robot *R) {
    if (R->millis > 10000) {
      // Increment a phase variable by a freq*dt
      phase += freq / ((float)gr::CONTROL_RATE);
    }
    for (int i=0; i<R->P.limbs_count; ++i) {
      R->setLimbPosition(i, desiredToePos(R, i, phase), kp, kd);
    }
  }

  inline Eigen::Vector3f desiredToePos(gr::Robot *R, int legi, float phase) {
    // Command a desired position where the y-position is the same as the abduction joint-hip offset
    float ab_hip_offs = R->P.limbs[legi].kinParams[3];
    float l0 = R->P.limbs[legi].kinParams[0];
    return Eigen::Vector3f(0, ab_hip_offs, -l0 * (0.8f - 0.3f*cosf(phase)));
  }

  // States
  float phase = 0.0f;

  void begin(const gr::Robot *R) {}
};

void debug() {
  // Robot status
  auto diag = robot.diagnostics();
  printf("%lx %lx\t", ~diag.health & diag.present, robot.P.version);
  for (int i=0; i<3; ++i) {
    printf("%.1f,", robot.S.diagnostics[i]);
  }
  printf("%.1f\t", robot.status.updateARate);

	for (int i = 0; i < 1; ++i) {
		auto pos = robot.getLimbPosition(i);
		printf("(%.2f,%.2f,%.2f)->", robot.getJointPosition(i, HIPJ), robot.getJointPosition(i, KNEEJ), robot.getJointPosition(i, ABDUCTIONJ));
		printf("(%.2f,%.2f,%.2f), ", pos.x(), pos.y(), pos.z());
	}
  printf("\n");
}

LowLevelLimbControl low_level_limb_control;

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

  robot.behaviors.clear();
  robot.behaviors.push_back(&low_level_limb_control);
  robot.requestBehavior(gr::Behav_CUSTOM, 1);

	return robot.begin();
}
