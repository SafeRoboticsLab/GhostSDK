/**
 * Copyright (C) Ghost Robotics - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Avik De <avik@ghostrobotics.io>
 */
#pragma once

#include <thread>
#include <atomic>
#include <Robot.hpp>
#include <JoyBase.h>

#if !defined(ARCH_mb8x)

namespace gr {

/**
 * @brief Update upstream communications (mavlink)
 * 
 * @param R 
 * @param useJoyMAV set false to not update joymavlink
 * @param joySim Pass local joystick (NULL if there is none)
 * @return int 
 */
int simUpstreamUpdate(Robot *R, bool useJoyMAV, JoyBase *joySim);

// Send a single packet
void simUpstreamSend(const uint8_t *txbuf, uint16_t len);

void simUpstreamRxThreadFun(Robot *R);

void simUpstreamCleanup();

class SimDebugTask {
public:
  SimDebugTask() {
    _thread = std::thread(&SimDebugTask::taskFun, this);
  }
  void stop() {
    _keepRunning.exchange(false);
    _thread.join(); // wait for exit
  }

  bool isRunning(){
    return _keepRunning;
  }
protected:
  std::atomic_bool _keepRunning{true};
  std::thread _thread;
  void taskFun();
};
  
}

#endif
