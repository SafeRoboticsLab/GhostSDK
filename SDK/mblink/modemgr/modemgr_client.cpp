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
#include "modemgr_client.hpp"
#include <string.h>

namespace gr {

ModeMgrClient::ModeMgrClient() {
	fieldFromStr_["turbo"] = Behav_TURBO;
	fieldFromStr_["blind_stairs"] = Behav_BLIND_STAIRS;
	fieldFromStr_["high_step"] = Behav_HIGH_STEP;
	fieldFromStr_["run"] = Behav_GAIT;
	fieldFromStr_["gait"] = Behav_GAIT;
	fieldFromStr_["dock"] = Behav_DOCK;
	fieldFromStr_["recovery"] = Behav_RECOVERY;
	fieldFromStr_["leap"] = Behav_LEAP;
	fieldFromStr_["roll_over"] = Behav_ROLL_OVER;
	fieldFromStr_["action"] = Behav_ACTION;
	fieldFromStr_["estop"] = Behav_ESTOP;
	fieldFromStr_["planner_en"] = Behav_PLANNER_EN;
	fieldFromStr_["planner_cmd"] = Behav_PLANNER_CMD;
	fieldFromStr_["planner_err"] = Behav_PLANNER_ERR;
	fieldFromStr_["custom"] = Behav_CUSTOM;
	fieldFromStr_["custom_index"] = Behav_CUSTOM_INDEX;
	fieldFromStr_["arm"] = Behav_ARM;
	fieldFromStr_["pose_rate"] = Behav_POSE_RATE;
	fieldFromStr_["leg_lock"] = Behav_LEG_LOCK;
	fieldFromStr_["status"] = Behav_STATUS;
	// Special request commands
	fieldFromStr_["normal"] = BehavReq_NORMAL;
	// selfcheck gets its own function
}

void ModeMgrClient::received(uint8_t bmode, uint32_t cmode) {
	mode_ = Mode_t(bmode, cmode);

	if (!receivedOnce_) {
		connect();
		receivedOnce_ = true;
	}
	
	// This is a hack
	size_t len = Behav_END;
	fieldVals_.resize(len);
	fieldVals_[Behav_EN] = decodeBehavior(&mode_, Behav_EN);
	fieldMap_["turbo"] = fieldVals_[Behav_TURBO] = decodeBehavior(&mode_, Behav_TURBO);
	fieldMap_["blind_stairs"] = fieldVals_[Behav_BLIND_STAIRS] = decodeBehavior(&mode_, Behav_BLIND_STAIRS);
	fieldMap_["high_step"] = fieldVals_[Behav_HIGH_STEP] = decodeBehavior(&mode_, Behav_HIGH_STEP);
	fieldMap_["run"] = fieldVals_[Behav_GAIT] = decodeBehavior(&mode_, Behav_GAIT);
	fieldMap_["gait"] = fieldMap_["run"];
	fieldMap_["dock"] = fieldVals_[Behav_DOCK] = decodeBehavior(&mode_, Behav_DOCK);
	fieldMap_["recovery"] = fieldVals_[Behav_RECOVERY] = decodeBehavior(&mode_, Behav_RECOVERY);
	fieldMap_["leap"] = fieldVals_[Behav_LEAP] = decodeBehavior(&mode_, Behav_LEAP);
	fieldMap_["roll_over"] = fieldVals_[Behav_ROLL_OVER] = decodeBehavior(&mode_, Behav_ROLL_OVER);
	fieldMap_["action"] = fieldVals_[Behav_ACTION] = decodeBehavior(&mode_, Behav_ACTION);
	fieldMap_["estop"] = fieldVals_[Behav_ESTOP] = decodeBehavior(&mode_, Behav_ESTOP);
	fieldMap_["planner_en"] = fieldVals_[Behav_PLANNER_EN] = decodeBehavior(&mode_, Behav_PLANNER_EN);
	fieldMap_["planner_cmd"] = fieldVals_[Behav_PLANNER_CMD] = decodeBehavior(&mode_, Behav_PLANNER_CMD);
	fieldMap_["planner_err"] = fieldVals_[Behav_PLANNER_ERR] = decodeBehavior(&mode_, Behav_PLANNER_ERR);
	fieldMap_["custom"] = fieldVals_[Behav_CUSTOM] = decodeBehavior(&mode_, Behav_CUSTOM);
	fieldMap_["custom_index"] = fieldVals_[Behav_CUSTOM_INDEX] = decodeBehavior(&mode_, Behav_CUSTOM_INDEX);
	fieldMap_["status"] = fieldVals_[Behav_STATUS] = decodeBehavior(&mode_, Behav_STATUS);
	fieldMap_["arm"] = fieldVals_[Behav_ARM] = decodeBehavior(&mode_, Behav_ARM);
	fieldMap_["pose_rate"] = fieldVals_[Behav_POSE_RATE] = decodeBehavior(&mode_, Behav_POSE_RATE);
	fieldMap_["leg_lock"] = fieldVals_[Behav_LEG_LOCK] = decodeBehavior(&mode_, Behav_LEG_LOCK);
}

FieldMap_t ModeMgrClient::getAll() const {
	return fieldMap_;
}

int32_t ModeMgrClient::get(const std::string &fieldstr) const {
	FieldNameToEnum_t::const_iterator iter = fieldFromStr_.find(fieldstr);

	return (iter != fieldFromStr_.end()) ? get(iter->second) : -1;
}

RequestTuple_t ModeMgrClient::set(const std::string &fieldstr, uint32_t val) const {
	FieldNameToEnum_t::const_iterator iter = fieldFromStr_.find(fieldstr);
	if (iter != fieldFromStr_.end())
		return set2(iter->second, val);
	else
		return std::make_tuple(0, 0);
}

RequestTuple_t ModeMgrClient::set2(BehavField_t field, uint32_t val) const {
	return std::make_tuple(field, val);;
}

RequestTuple_t ModeMgrClient::selfCheck(int16_t action, int16_t param) const {
	// Pack into a request
	static uint32_t packedRequest;
	static SelfCheckCmd selfCheckCmd;
	// Cast to signed here
	selfCheckCmd.action = action;
	selfCheckCmd.param = param;
	memcpy(&packedRequest, &selfCheckCmd, 4);
	return set2(BehavReq_SELF_CHECK, packedRequest);
}

} // namespace gr
