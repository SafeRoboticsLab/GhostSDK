/**
MIT License (modified)

Copyright (c) 2021 Ghost Robotics
Authors:
Avik De <avik@ghostrobotics.io>

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
#include "modemgr.hpp"

// https://docs.google.com/spreadsheets/d/11Q8HBxHXZNh290qA0E7Ryg2Pj_cHm3hP7-OCjtFwf90/edit#gid=0

namespace gr {


void encodeBehavior(Mode_t *mode, BehavField_t field, uint32_t value) {
	switch (field) {
		case Behav_EN:
			bitWrite(mode->cmode, 16, value);
			break;
		case Behav_TURBO:
			bitWrite(mode->cmode, 2, value);
			break;
		case Behav_BLIND_STAIRS:
			bitWrite(mode->cmode, 3, value);
			break;
		case Behav_HIGH_STEP:
			bitWrite(mode->cmode, 4, value);
			break;
		case Behav_GAIT:
			mode->cmode &= ~(0b111UL << 5); // clear what is there initially
			mode->cmode |= (value & 0b111UL) << 5;
			break;
		case Behav_DOCK:
			bitWrite(mode->cmode, 8, value);
			break;
		case Behav_RECOVERY:
			mode->cmode &= ~(0b11UL << 9); // clear what is there initially
			mode->cmode |= (value & 0b11UL) << 9;
			break;
		case Behav_LEAP:
			bitWrite(mode->cmode, 11, value);
			break;
		case Behav_ROLL_OVER:
			bitWrite(mode->cmode, 12, value);
			break;
		case Behav_ARM:
			bitWrite(mode->cmode, 13, value);
			break;
		case Behav_POSE_RATE:
			bitWrite(mode->cmode, 14, value);
			break;
		case Behav_LEG_LOCK:
			bitWrite(mode->cmode, 15, value);
			break;
		case Behav_ACTION:
			mode->cmode &= ~(0b11UL); // clear what is there initially
			mode->cmode |= (value & 0b11UL);
			break;
		case Behav_ESTOP:
			bitWrite(mode->cmode, 18, value);
			break;
		case Behav_PLANNER_EN:
			bitWrite(mode->cmode, 19, value);
			break;
		case Behav_PLANNER_CMD:
			mode->cmode &= ~(0b11UL << 20); // clear what is there initially
			mode->cmode |= (value & 0b11UL) << 20;
			break;
		case Behav_PLANNER_ERR:
			// Does not make sense to request an error
			break;
		case Behav_CUSTOM:
			bitWrite(mode->cmode, 24, value);
			break;
		case Behav_CUSTOM_INDEX:
			mode->cmode &= ~(0b111UL << 25); // clear what is there initially
			mode->cmode |= (value & 0b111UL) << 25;
			break;
		case Behav_STATUS:
			mode->cmode &= ~(0b1111UL << 28); // clear what is there initially
			mode->cmode |= (value & 0b1111UL) << 28;
			break;
		default:
			break;
	}
}

uint32_t decodeBehavior(const Mode_t *mode, BehavField_t field) {
	switch (field) {
		case Behav_EN:
			return bitRead(mode->cmode, 16);
		case Behav_TURBO:
			return bitRead(mode->cmode, 2);
		case Behav_BLIND_STAIRS:
			return bitRead(mode->cmode, 3);
		case Behav_HIGH_STEP:
			return bitRead(mode->cmode, 4);
		case Behav_GAIT:
			return (mode->cmode >> 5) & 0b111UL;
		case Behav_DOCK:
			return bitRead(mode->cmode, 8);
		case Behav_RECOVERY:
			return (mode->cmode >> 9) & 0b11UL;
		case Behav_LEAP:
			return bitRead(mode->cmode, 11);
		case Behav_ROLL_OVER:
			return bitRead(mode->cmode, 12);
		case Behav_ARM:
			return bitRead(mode->cmode, 13);
		case Behav_POSE_RATE:
			return bitRead(mode->cmode, 14);
		case Behav_LEG_LOCK:
			return bitRead(mode->cmode, 15);
		case Behav_ACTION:
			return (mode->cmode) & 0b11UL;
		case Behav_ESTOP:
			return bitRead(mode->cmode, 18);
		case Behav_PLANNER_EN:
			return bitRead(mode->cmode, 19);
		case Behav_PLANNER_CMD:
			return (mode->cmode >> 20) & 0b11UL;
		case Behav_PLANNER_ERR:
			return (mode->cmode >> 22) & 0b11UL;
		case Behav_CUSTOM:
			return bitRead(mode->cmode, 24);
		case Behav_CUSTOM_INDEX:
			return (mode->cmode >> 25) & 0b111UL;
		case Behav_STATUS:
			return (mode->cmode >> 28) & 0b1111UL;
		default:
			return 0;
	}
}

} // namespace gr
