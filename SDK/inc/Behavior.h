/*
 * Copyright (C) Ghost Robotics - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Avik De <avik@ghostrobotics.io>
 */
#pragma once

#include "Peripheral.h"
#include <stdint.h>
#include <modemgr.hpp>

#if defined(__clang__)

#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)

#endif

namespace gr {

/** @addtogroup Behavior Behavior
 *  @{
 */

/**
 * @brief Abstract base class for implementing behaviors. Only one behavior is active
 * after initialization.
 */
class Behavior : public Peripheral {
public:
  /**
   * @brief Called when a stop is requested by a BehaviorCmd
   */
  virtual void end() {}

  /**
   * @brief Signal a change in the BehavField_t
   * 
   * @param field For Behav_EN, begin() or end() are called. This can be Behav_TURBO, Behav_BLIND_STAIRS, Behav_HIGH_STEP, Behav_GAIT, Behav_DOCK, Behav_PLANNER_EN.
   * @param newval 
   */
  virtual void signal(gr::Robot *R, BehavField_t field, uint32_t newval) {}

  /**
   * @brief The behavior can indicate if it is done with the transition requested above
   * 
   * @param field Can be Behav_EN, Behav_TURBO, Behav_BLIND_STAIRS, Behav_HIGH_STEP, Behav_GAIT, Behav_DOCK, Behav_PLANNER_EN.
   * @return true Transition is completed (proceed with sequential composition)
   * @return false Should wait.
   */
  virtual bool completed(gr::Robot *R, BehavField_t field, uint32_t val, uint32_t newval) { return true; }
};

/** @} */ // end of addtogroup

#if defined(__clang__)

#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)

#endif
  
}

