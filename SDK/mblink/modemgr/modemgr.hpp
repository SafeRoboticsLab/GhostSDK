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
#pragma once
#include <stdint.h>

namespace gr {

/** @addtogroup Modes Behavior modes
 * @brief Robot ("mainboard") discrete modes
 *  @{
 */

#ifndef bitRead
#define bitRead(value, bit) (((value) >> (bit)) & 0x01)
#define bitSet(value, bit) ((value) |= (1UL << (bit)))
#define bitClear(value, bit) ((value) &= ~(1UL << (bit)))
#define bitWrite(value, bit, bitvalue) (bitvalue ? bitSet(value, bit) : bitClear(value, bit))
#endif

#pragma pack(push,1)
struct SelfCheckCmd {
	int16_t action;
	int16_t param;
}; // 32 bit in cmode

struct HmapHdr_t {
	uint8_t type; //1
	uint8_t res_mm; //2
	int16_t xstart_cm; //4
	uint16_t num_points; //6
	uint8_t mb_rate, cam_rate; //8
}; // must go in 64 bits = 8 bytes

struct ObstHdr_t {
	uint8_t type; //1
	uint8_t rad_cm; //2
	uint8_t num_obsts; //3
}; // must go in 64 bits = 8 bytes
#pragma pack(pop)

/**
 * @brief IDs of debug float arrays sent from/to the mainboard using mavlink
 */
enum DebugFloatArrayID_t {
	DFAID_JOINT_STATES = 0,
	DFAID_HMAP_Z = 1,
	DFAID_HMAP_X = 2,
	DFAID_HMAP_DZDX = 3,
	DFAID_HMAP_DZDX_PADDING = 4,
	DFAID_PLANNER_LOG = 5,
	DFAID_OBST_CENTER = 6,
};

/**
 * @brief This struct contains the encoded mode. If we change the encoding in the future, only this file
 * and modemgr.cpp should need to be changed in the SDK core.
 */
struct Mode_t {
	uint8_t bmode;
	uint32_t cmode;
	Mode_t() : bmode(0), cmode(0) {}
	Mode_t(uint8_t bmode, uint32_t cmode) : bmode(bmode), cmode(cmode) {}
	void operator = (const Mode_t &rhs) { bmode = rhs.bmode; cmode = rhs.cmode; }
	friend bool operator == (const Mode_t &lhs, const Mode_t &rhs) { return lhs.bmode == rhs.bmode && lhs.cmode == rhs.cmode; }
	friend bool operator != (const Mode_t &lhs, const Mode_t &rhs) { return !(lhs == rhs); }
};

/**
 * @brief Behavior fields contained in the robot discrete mode.
 */
enum BehavField_t {
	// These are not parts of the mode, but instead special requests
	BehavReq_NONE = 0, ///< Reserved
	BehavReq_NORMAL, ///< A special request to go back to normal walk. String: "normal"
	BehavReq_SELF_CHECK, ///< A special request to pass hardware self check commands. String: "self_check"
	// These are robot operation modes
	Behav_EN, ///< deprecated
	Behav_TURBO, ///< Turbo mode (0 or 1). String: "turbo"
	Behav_BLIND_STAIRS, ///< Blind stair mode (0 or 1). String: "blind_stairs"
	Behav_HIGH_STEP, ///< High step mode for curbs etc. (0 or 1). String: "high_step"
	Behav_GAIT,  ///< 0=walk, 1=run, 2=hill mode. String: "gait"
	Behav_DOCK, ///< 1=dock mode. Affects walking and standing. String: "dock"
	Behav_RECOVERY, ///< 1 indicates robot is in recovery. User should not set. String: "recovery"
	Behav_LEAP, ///< unused
	Behav_ROLL_OVER, ///< Setting 1 makes the robot roll over. String: "roll_over"
	Behav_ACTION, ///< 0=sit, 1=stand/lookaround, 2=walk/move. String: "action"
	Behav_ESTOP, ///< 1 means e-stopped. String: "estop"
	Behav_PLANNER_EN, ///< 1 means planner is enabled. String: "planner_en"
	Behav_PLANNER_CMD, ///< 0=waypoints mode, 1=planner stepping curbs etc., 2=stairs 242, 3=stairs
	Behav_PLANNER_ERR, ///< unused
	Behav_CUSTOM, ///< 1 indicates a custom behavior is being used. Must be set to use custom behaviors. String "custom"
	Behav_CUSTOM_INDEX, ///< index into the behavior vector. Can be changed to switched to a different custom behavior. String: "custom_index"
	Behav_STATUS, ///< 
	Behav_ARM, ///< 1 when an arm is attached. String: "arm"
	Behav_POSE_RATE, ///< 1 when pose rate mode is enabled. Affects lookaround and walking pose. String: "pose_rate"
	Behav_LEG_LOCK, ///< 1 to lock the legs into a carrying-friendly position. String: "leg_lock"

	Behav_END /// NOTE: new behaviors must be above this!
};

/**
 * @brief Encode into a packed Mode_t a new value for some field
 * 
 * @param mode The packed mode which is written to and updated
 * @param field The field to write
 * @param value The value to write to the field
 */
void encodeBehavior(Mode_t *mode, BehavField_t field, uint32_t value);

/**
 * @brief Decode from a packed Mode_t the value for some field
 * 
 * @param mode The packed mode which is read from
 * @param field The field to read
 * @return uint32_t The value that was in this field
 */
uint32_t decodeBehavior(const Mode_t *mode, BehavField_t field);

/** @} */ // end of addtogroup

}
