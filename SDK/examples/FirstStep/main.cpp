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

void debug() {
  // Robot status
  auto diag = robot.diagnostics();
  printf("%x %x\t", ~diag.health & diag.present, robot.P.version);
  for (int i=0; i<3; ++i) {
    printf("%.1f,", robot.S.diagnostics[i]);
  }
  printf("%.1f\t", robot.status.updateARate);

	for (int i = 0; i < 1; ++i) {
		auto pos = robot.getLimbPosition(i);
		printf("(%.2f,%.2f,%.2f)->", robot.getJointPosition(i, HIPJ), robot.getJointPosition(i, KNEEJ), robot.getJointPosition(i, ABDUCTIONJ));
		printf("(%.2f,%.2f,%.2f), ", pos.x(), pos.y(), pos.z());
	}

  // IMU
  const auto *I = &robot.S.imu;
  // printf("\teul\t%.2f,%.2f,%.2f,", I->euler.x, I->euler.y, I->euler.z);
  // printf("\tangrate\t%.2f,%.2f,%.2f,", I->angular_velocity.x, I->angular_velocity.y,I->angular_velocity.z);
  printf("\tacc\t%.2f,%.2f,%.2f,", I->linear_acceleration.x, I->linear_acceleration.y, I->linear_acceleration.z);

  printf("\n");
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

  // gr::setDebugRate(0); // Set 0 to disable debug printing, otherwise set the rate in Hz

	return robot.begin();
}
