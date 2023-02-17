/*
 * Copyright (C) Ghost Robotics - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Avik De <avik@ghostrobotics.io>
 */
#pragma once

#include "Robot.hpp"

namespace gr {

/** @addtogroup bsp SDK implementation "board support package"
 * @brief Set of functions each platform needs to implement for the SDK
 *  @{
 */

/**
 * @brief Initialize various data fields (called from Robot::init())
 * 
 * @param pRobot Pointer to the robot
 * @param argc 
 * @param argv 
 */
void bspInit(Robot *pRobot, int argc, char *argv[]);

/**
 * @brief Actually start various tasks and spin. Called from Robot::begin() typically at the end of main().
 * 
 * @param pRobot Pointer to the robot.
 */
void bspBegin(Robot *pRobot);

/**
 * @brief Get the current time in milliseconds
 * 
 * @return uint32_t Time since init in milliseconds
 */
uint32_t bspGetMillis();

/**
 * @brief Get joint information and populate JointState
 * 
 * @param P Robot parameters (input argument)
 * @param i Joint index
 * @param rawState (output) JointState which should be populated
 * @return int 0 if successful, <0 if not
 */
int bspJointGetRaw(const RobotParams *P, int i, JointState *rawState);

/**
 * @brief Send a joint command to the hardware or simulation dynamics
 * 
 * @param P Robot parameters (input argument)
 * @param i Joint index
 * @param mode Joint control mode from JointMode enum
 * @param setpoint Argument for the  control mode
 * @return int 0 if successful, <0 if not
 */
int bspJointSendCommand(const RobotParams *P, int i, uint16_t mode, float setpoint);

/**
 * @brief Get the external force on a leg using sensors, if available
 * 
 * @param P Robot parameters (input argument)
 * @param i Joint index
 * @param endEffForce (output) end effector force
 * @return true Data is available and valid
 * @return false External force sensors not present, or data not available
 */
bool bspGetExternalForce(const RobotParams *P, int i, Eigen::Vector3f &endEffForce);


/**
 * @brief Actuation update (critical, hard real-time)
 * @details Should be called by the implementation
 */
void sdkUpdateA(Robot *r);
/**
 * @brief Behavior and other task update (less critical, but still should be called as often as possition)
 * @details Should be called by the implementation
 */
void sdkUpdateB(Robot *r);

/** @} */ // end of addtogroup

}
