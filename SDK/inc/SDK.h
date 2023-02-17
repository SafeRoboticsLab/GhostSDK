/*
 * Copyright (C) Ghost Robotics - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Avik De <avik@ghostrobotics.io>
 */
#pragma once

#include "Robot.hpp"
#include "bsp.hpp"
#include "SimSetup.hpp"
#include "MAVLinkUser.hpp"
#include "HardwareSetup.h"

namespace gr {
	
/** @addtogroup SDK Miscellaneous SDK functions
 *  @{
 */

/**
 * Defined separately for each platform, in units of Hz. The user should not set this but can read it. This can also be read via mblink res["debug_timings"].
 */
extern "C" int CONTROL_RATE;
/**
 * Maximum number of limbs
 */
#define MAX_LIMB_COUNT              	4

/**
 * @brief System clock time (in microseconds)
 */
uint32_t clockTimeUS();

/**
 * @brief Set the rate at which the debug() user function is called
 * @param hz Rate in Hz; set 0 to disable debug printing
 */
void setDebugRate(int hz);

/** @} */ // end of addtogroup

}
