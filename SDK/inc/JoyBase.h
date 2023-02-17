/**
 * Copyright (C) Ghost Robotics - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Avik De <avik@ghostrobotics.io>
 */
#pragma once

#include "Peripheral.h"
#include "simple.pb.h"
#include <SMath.h>
#include <modemgr.hpp>

namespace gr {

/**
	 * @brief Updates a float to desired within rate limits
	 * 
	 * @param current state to update
	 * @param desired new desired
	 * @param rateLimit limit in units of /second
	 * @param rateHz
	 */
void rateLimitedUpdate(float &current, float desired, float rateLimit, float rateHz);

class JoyBase : public Peripheral {
protected:
	float JOY_SPEED_SENS = 1.0f;//m/s. 0.6 (low), 0.8 (medium), 1.5 (high)
	float JOY_YAW_SENS = 1.6f;//rad/s.

	Twist normTwist_; // temporary storage before scaling
	Pose normPose_;

	// State for filtering
	Eigen::Vector3f poseDes, linearDes, angularDes;

	bool fixedUpdateRate = false;
	float fixedRateHz = 0;

public:
	/**
	 * @brief Init filters
	 * 
	 */
	void init();

	// Behavior requests 
	virtual void convertBehavior(gr::Robot* /*R*/) {}
	
	// From peripheral
	virtual void begin(const Robot *R) = 0;
	virtual void update(Robot *R) = 0;
	/**
	 * @brief This should output a *normalized* BehaviorCmd. The SDK will then scale the inputs according the robot's abilities.
	 * 
	 * @param b 
	 */
	virtual void toNormalizedBehaviorCmd(Robot *R, Twist *normTwist, Pose *normPose) = 0;

	/**
	 * @brief This will be called by the SDK to (a) get the normalized behavior cmd, and (b) scale the inputs
	 * 
	 * @param b 
	 * @param scalePose Scales lookaround commands (set false for direct control in radians)
	 */
	void toBehaviorCmd(Robot *R, BehaviorCmd *b, bool scalePose);

	/**
	 * @brief Set joystick sensitivity
	 * 
	 * @param speedSens Joystick axis at max range maps to +/- speedSens m/s
	 * @param yawSens Joystick axis at max range maps to +/- yawSens rad/s
	 */
	void setSensitivity(float speedSens, float yawSens) {
		JOY_SPEED_SENS = speedSens;
		JOY_YAW_SENS = yawSens;
	}

	void setUpdateRate(float rateHz) {
		fixedUpdateRate = true;
		fixedRateHz = rateHz;
	}

	virtual bool timedOut(const gr::Robot* /*R*/) const { return false; }

	virtual bool requestsOverride(const gr::Robot* /*R*/) { return false; }

	virtual ~JoyBase() {}
};
	
}
