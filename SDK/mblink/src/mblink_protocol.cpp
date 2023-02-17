/**
MIT License (modified)

Copyright (c) 2021 Ghost Robotics
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
#include <mblink/mblink_protocol.hpp>
#include <iostream>

namespace gr
{

template<size_t N>
using EigMapN = Eigen::Map<const Eigen::Matrix<float, N, 1>>;

void MBLinkProtocol::initRxData() {
	// param_value - only when received
	// Commands
	rxdata["se2twist_des"].resize(3);
	rxdata["joy_twist"].resize(4);
	rxdata["joy_buttons"].resize(8);
	// Misc
	rxdata["debug_legH"].resize(3);
	rxdata["user"].resize(10);
	// Joints
	rxdata["joint_cmd"].resize(12);
	rxdata["joint_position"].resize(12);
	rxdata["joint_velocity"].resize(12);
	rxdata["joint_current"].resize(12);
	rxdata["joint_temperature"].resize(12);
	rxdata["joint_voltage"].resize(12);
	rxdata["joint_residual"].resize(8);
	rxdata["joint_status"].resize(12);
	// Swing/contact states and behavior
	rxdata["phase"].resize(4);
	rxdata["behavior"].resize(3);
	rxdata["contacts"].resize(4);
	rxdata["swing_mode"].resize(4);
	rxdata["mode"].resize(2);
	// IMU
	rxdata["imu_euler"].resize(3);
	rxdata["imu_linear_acceleration"].resize(3);
	rxdata["imu_angular_velocity"].resize(3);
	// State estimates
	rxdata["twist_linear"].resize(3);
	rxdata["z_rel"].resize(1);
	rxdata["slope_est"].resize(2);
	rxdata["y"].resize(21);// last element is time
	// Debugging
	rxdata["voltage"].resize(2);
	rxdata["debug_timings"].resize(4);
	rxdata["diagnostics"].resize(2);
	rxdata["version"].resize(1); // log version

	// ctrldata detect size from data
}

int MBLinkProtocol::setModeRaw(uint8_t bMode, uint32_t cMode)
{
	return queueMessage(mavlink_msg_set_mode_pack(systemId, componentId, &msg, 1, bMode, cMode));
}

int MBLinkProtocol::sendControl(const Eigen::VectorXf &uin)
{
	Eigen::VectorXf u = uin;
	if (u.size() < 12)
	{
		u.conservativeResizeLike(Eigen::Matrix<float, 12, 1>::Zero());
	}
	return queueMessage(mavlink_msg_landing_target_pack(systemId, componentId, &msg, 
		this->nsteps, 0, 
		0, //< "frame" unused for now
		u[0], u[1], u[2], //< se2twist
		u[3], u[4], u[5], //< r1
		u[6], u[7], u.tail<4>().data(), //< r2, poseZRP
		0, 0));
}

int MBLinkProtocol::sendPose(const Eigen::Vector3f &position, const Eigen::Vector3f &orientation)
{
	return queueMessage(mavlink_msg_attitude_pack(systemId, componentId, &msg, 0, 
		orientation.x(), orientation.y(), orientation.z(), 
		0, position.z(), 0)); // FIXME: only z is used
}

int MBLinkProtocol::requestMode(const std::string &fieldName, uint32_t val)
{
	auto tosend = mmgr.set(fieldName, val);
	return setModeRaw(std::get<0>(tosend), std::get<1>(tosend));
}

int MBLinkProtocol::requestParam(const std::string &param_id)
{
	// to read, clear the param value (can check its size for reception)
	paramMutex.lock();
	rxdata["param_value"].resize(0);
	paramMutex.unlock();
	// http://www.cplusplus.com/reference/cstring/strncpy/ If the end of the source C string (which is signaled by a null-character) is found before num characters have been copied, destination is padded with zeros until a total of num characters have been written to it.
	strncpy(paramIdBuf, param_id.c_str(), 17);
	return queueMessage(mavlink_msg_param_request_read_pack(systemId, componentId, &msg, 0, 0, paramIdBuf, -1));
}

int MBLinkProtocol::setParam(const std::string &param_id, float value)
{
	strncpy(paramIdBuf, param_id.c_str(), 17);
	return queueMessage(mavlink_msg_param_set_pack(systemId, componentId, &msg, 0, 0, paramIdBuf, value, MAV_PARAM_TYPE_REAL32));
}

int MBLinkProtocol::sendPlannerParam(const std::string &name, const Eigen::VectorXf &val, bool write) {
	strncpy(param_name_res_req_, name.c_str(), 120);
	return queueMessage(mavlink_msg_resource_request_pack(systemId, componentId, &msg, 0, write ? 1 : 0, (const uint8_t *)param_name_res_req_, (uint8_t)val.size(), (const uint8_t *)val.data()));
}

int MBLinkProtocol::sendGoal(uint8_t goal_type, uint8_t frame, Eigen::VectorXf &goal) {
	uint16_t seq = (uint16_t)goal.size();
	goal.conservativeResizeLike(Eigen::Matrix<float, 7, 1>::Zero());
	return queueMessage(mavlink_msg_mission_item_pack(systemId, componentId, &msg, 0, 0, seq, frame, 0, 0, 0, goal[0], goal[1], goal[2], goal[3], goal[4], goal[5], goal[6], goal_type));
}

int MBLinkProtocol::sendToSim(uint32_t type, uint16_t start, uint16_t stop, const Eigen::Matrix<float, 58, 1> &floatArray)
{
	// create the header
	static uint64_t time_usec;
	MessageFromPlannerHeader_t *phdr = (MessageFromPlannerHeader_t *)&time_usec;
	phdr->type = type;
	phdr->start = start;
	phdr->stop = stop;
	// uint8_t *ptime_usec = (uint8_t *)&time_usec;
	// memcpy(&ptime_usec[0], &type, 4);
	// memcpy(&ptime_usec[4], &start, 2);
	// memcpy(&ptime_usec[6], &stop, 2);
	// pack
	return queueMessage(mavlink_msg_debug_float_array_pack(systemId, componentId, &msg, time_usec, "", DEBUG_FLOAT_ARRAY_ID_FROM_PLANNER, floatArray.data()));
}

int MBLinkProtocol::sendUser(const Eigen::Matrix<float, 58, 1> &data)
{
	return queueMessage(mavlink_msg_debug_float_array_pack(systemId, componentId, &msg, 0, "", DEBUG_FLOAT_ARRAY_ID_USER, data.data()));
}

std::string MBLinkProtocol::getPackedBytes()
{
	std::string ret;
	uint16_t nbytes = mavlink_msg_to_send_buffer(txbytes, &msg);
	ret.resize(nbytes);
	std::copy(&txbytes[0], &txbytes[nbytes], ret.begin());
	return ret;
}

bool MBLinkProtocol::unpack(const mavlink_message_t *msg)
{
	int _offs = 0; // offset into arrays when needed

	switch (msg->msgid)
	{
	case MAVLINK_MSG_ID_WIND_COV:
		mavlink_msg_wind_cov_decode(msg, &wind_cov);
		rxdata["slope_est"] = Eigen::Vector2f(wind_cov.wind_x, wind_cov.wind_y);
		rxdata["se2twist_des"] = Eigen::Vector3f(wind_cov.wind_z, wind_cov.var_horiz, wind_cov.var_vert);
		rxdata["debug_legH"] = Eigen::Vector3f(wind_cov.wind_alt, wind_cov.horiz_accuracy, wind_cov.vert_accuracy);

		rxdata["user"].head<2>() = EigMapN<2>((float *)&wind_cov.time_usec);
		rxdata["user"].tail<8>() << 
			wind_cov.wind_x, wind_cov.wind_y, wind_cov.wind_z, 
			wind_cov.var_horiz, wind_cov.var_vert, wind_cov.wind_alt, 
			wind_cov.horiz_accuracy, wind_cov.vert_accuracy;

		// This is the last one sent by the mb
		constructStateVector();
		return true;

	case MAVLINK_MSG_ID_LANDING_TARGET:
		mavlink_msg_landing_target_decode(msg, &landing_target);
		if (landing_target.frame == LANDING_TARGET_FRAME_JOINT_CMD)
		{
			rxdata["joint_cmd"] << 
				landing_target.angle_x, landing_target.angle_y, 
				landing_target.distance, landing_target.size_x,
				landing_target.size_y, landing_target.x,
				landing_target.y, landing_target.z,
				EigMapN<4>(landing_target.q);
		}
		return true;

	case MAVLINK_MSG_ID_MEMORY_VECT:
		mavlink_msg_memory_vect_decode(msg, &memory_vect);
		_offs = 0;
		memcpy(&behaviorId, &memory_vect.value[_offs], 4);
		_offs += 4; //4
		memcpy(&behaviorMode, &memory_vect.value[_offs], 4);
		_offs += 4; //8
		memcpy(contacts, &memory_vect.value[_offs], 4);
		_offs += 4; //12
		for (int i = 0; i < 4; ++i)
		{
			memcpy(&swingMode[i], &memory_vect.value[_offs], 1);
			_offs += 1;
		} //16
		rxdata["phase"] = EigMapN<4>((float *)&memory_vect.value[_offs]); //32
		rxdata["behavior"] = Eigen::Vector3f((float)behaviorId, (float)behaviorMode, (float)status);
		rxdata["contacts"] = Eigen::Vector4f(contacts[0], contacts[1], contacts[2], contacts[3]);
		rxdata["swing_mode"] = Eigen::Vector4f(swingMode[0], swingMode[1], swingMode[2], swingMode[3]);

		return true;

	case MAVLINK_MSG_ID_CONTROL_SYSTEM_STATE:
		mavlink_msg_control_system_state_decode(msg, &control_system_state);
		// Number of steps
		this->nsteps = (int)control_system_state.time_usec;
		// Center of mass position relative to the toe centroid
		pcom = Eigen::Vector3f(control_system_state.x_pos, control_system_state.y_pos, control_system_state.z_pos);
		// IMU Euler is in the first 3
		rxdata["imu_euler"] = EigMapN<3>(control_system_state.q);
		rxdata["imu_linear_acceleration"] = Eigen::Vector3f(control_system_state.x_acc, control_system_state.y_acc, control_system_state.z_acc);
		rxdata["twist_linear"] = Eigen::Vector3f(control_system_state.x_vel, control_system_state.y_vel, control_system_state.z_vel);
		rxdata["imu_angular_velocity"] = Eigen::Vector3f(control_system_state.roll_rate, control_system_state.pitch_rate, control_system_state.yaw_rate);
		ptoe << EigMapN<3>(control_system_state.vel_variance), EigMapN<3>(control_system_state.pos_variance);
		rxdata["z_rel"] << control_system_state.airspeed;
		unpackQuat3((const uint32_t *)&control_system_state.q[3]);

		return true;

	case MAVLINK_MSG_ID_DEBUG_FLOAT_ARRAY:
		mavlink_msg_debug_float_array_decode(msg, &debug_float_array);
		// std::cout << "received dfa " << debug_float_array.array_id << std::endl;
		if (debug_float_array.array_id == gr::DFAID_JOINT_STATES && msgFromSDK(msg)) {
			// This is actually millis
			this->t = debug_float_array.time_usec * 1e-3f;

			if (this->t > lastTime)
			{
				// msg.data contains LANIx12 (=48 floats) + joint residuals (x8 floats)
				LANI *lani = (LANI *)debug_float_array.data;

				for (int j = 0; j < 12; ++j)
				{
					rxdata["joint_position"][j] = lani[j].pos;
					rxdata["joint_velocity"][j] = lani[j].vel;
					rxdata["joint_current"][j] = lani[j].curr * 1e-2f;
					rxdata["joint_temperature"][j] = lani[j].temperature * 1e-2f;
					rxdata["joint_voltage"][j] = lani[j].voltage * 1e-3f;
					rxdata["joint_status"][j] = lani[j].param4;
				}

				rxdata["joint_residual"] = EigMapN<8>(&debug_float_array.data[48]);

				this->mvoltage = rxdata["joint_voltage"].mean();
			}

			return true;
		} else if (debug_float_array.array_id == gr::DFAID_OBST_CENTER && msgFromPlanner(msg)) {
			ctrldata_["obst_centers"] = EigMapN<58>(&debug_float_array.data[0]);
			const ObstHdr_t *phdr = (const ObstHdr_t *)&debug_float_array.time_usec;
			ctrldata_["obst_num_rad"].resize(2);
			ctrldata_["obst_num_rad"] << phdr->num_obsts, 0.01f*phdr->rad_cm;
			return false; // do not count towards mb rate
		} else if (debug_float_array.array_id == gr::DFAID_PLANNER_LOG && msgFromPlanner(msg)) {
			ctrldata_["pl_t"] = EigMapN<1>(&debug_float_array.data[0]);
			ctrldata_["pl_y"] = EigMapN<20>(&debug_float_array.data[1]);
			ctrldata_["u"] = EigMapN<12>(&debug_float_array.data[21]);
			ctrldata_["goal"] = EigMapN<3>(&debug_float_array.data[33]);
			ctrldata_["pjnom"] = EigMapN<6>(&debug_float_array.data[36]);
			ctrldata_["pjdes"] = EigMapN<6>(&debug_float_array.data[42]);
			ctrldata_["ter"] = EigMapN<2>(&debug_float_array.data[48]);
			ctrldata_["dock"] = EigMapN<6>(&debug_float_array.data[50]);
			ctrldata_["insp"] = EigMapN<2>(&debug_float_array.data[56]);
			// This is the last one in the UDP packet
			accumulateData(rx_accum_, ctrldata_);
			last_pl_time_ = t;
			return false; // do not count towards mb rate
		} else {
			return false;
		}

	case MAVLINK_MSG_ID_SYS_STATUS:
		mavlink_msg_sys_status_decode(msg, &sys_status);
		rxdata["voltage"] = Eigen::Vector2f(sys_status.voltage_battery, this->mvoltage);
		this->status = sys_status.errors_comm;
		rxdata["debug_timings"] = Eigen::Vector4f(sys_status.errors_count1, sys_status.errors_count2, sys_status.errors_count3, sys_status.errors_count4);
		rxdata["diagnostics"] << (sys_status.onboard_control_sensors_present & ~sys_status.onboard_control_sensors_health), (sys_status.onboard_control_sensors_present & ~sys_status.onboard_control_sensors_enabled);
		return true;

	case MAVLINK_MSG_ID_MANUAL_CONTROL:
		mavlink_msg_manual_control_decode(msg, &manual_control);

		// Joy twists
		rxdata["joy_twist"].x() = manual_control.x / (float)INT16_MAX;
		rxdata["joy_twist"].x() *= (rxdata["joy_twist"].x() > 0) ? MANCON_SCALE_XP : MANCON_SCALE_XM;
		// LinY
		rxdata["joy_twist"].y() = MANCON_SCALE_Y * manual_control.y / (float)INT16_MAX;
		// AngZ
		rxdata["joy_twist"][2] = MANCON_SCALE_RZ * manual_control.r / (float)INT16_MAX;
		// Height
		rxdata["joy_twist"][3] = (manual_control.z / (float)INT16_MAX) * 0.5f * (MANCON_Z_MAX - MANCON_Z_MIN) + 0.5f * (MANCON_Z_MAX + MANCON_Z_MIN);

		// Buttons. Each pair of bits is a button
		for (int i = 0; i < 8; ++i) {
			rxdata["joy_buttons"][i] = (float)(((manual_control.buttons) >> (2*i)) & 0b11);
		}
		return true;

	case MAVLINK_MSG_ID_PARAM_VALUE:
		mavlink_msg_param_value_decode(msg, &param_value);
		paramMutex.lock();
		rxdata["param_value"].resize(1);
		rxdata["param_value"] << param_value.param_value;
		paramMutex.unlock();
		paramcv.notify_one();
		return true;

	case MAVLINK_MSG_ID_HEARTBEAT:
		mavlink_msg_heartbeat_decode(msg, &heartbeat);
		modeMutex.lock();
		rxdata["mode"] = Eigen::Vector2f(heartbeat.base_mode, heartbeat.custom_mode);
		modeMutex.unlock();
		mmgr.received(heartbeat.base_mode, heartbeat.custom_mode);
		modecv.notify_one();
		return true;

	case MAVLINK_MSG_ID_AUTOPILOT_VERSION:
		mavlink_msg_autopilot_version_decode(msg, &autopilot_version_);
		rxdata["version"] << autopilot_version_.uid;
		return true;

	case MAVLINK_MSG_ID_RESOURCE_REQUEST: // planner param setting response
		mavlink_msg_resource_request_decode(msg, &resource_request);
		paramMutex.lock();
		rxdata["param_value"] = Eigen::Map<const Eigen::VectorXf>((const float *)resource_request.storage, resource_request.transfer_type);
		paramMutex.unlock();
		paramcv.notify_one();

	default:
		break;
	}
	return false;
}

void MBLinkProtocol::unpackQuat3(const uint32_t *packed) {
	int16_t swphasei;
	memcpy(&swphasei, packed, 2);
	sw_phase_ = 1e-4f * swphasei;

	// Unpack flight leg
	fl_ = ((*packed) >> 20) & 0b1;
	int fl_neg = ((*packed) >> 22) & 0b1;
	if (fl_neg != 0)
		fl_ = -1;
	
	stance_leg_03_ = ((*packed) >> 21) & 0b1;
}

int MBLinkProtocol::parse(const char *bytes, size_t length)
{
	int retval = 0;
	for (size_t i = 0; i < length; ++i)
	{
		if (mavlink_parse_char(rxchan, bytes[i], &rxmsg, &rxstatus) && (msgFromPlanner(&rxmsg) || msgFromSDK(&rxmsg))) {
			retval += unpack(&rxmsg);
		}
	}
	return retval;
}

void MBLinkProtocol::constructStateVector()
{
	rxdata["y"] << pcom, rxdata["imu_euler"], ptoe, rxdata["twist_linear"], rxdata["imu_angular_velocity"], (float)fl_, sw_phase_, t;

	accumulateData(rx_accum_, rxdata);
	// flag that there is new data
	newRxData.exchange(true);
	cv.notify_one();
}

void MBLinkProtocol::newPlannerUDPMsg(const char *msg, uint16_t len) {
	// Interpret the rest as a matrix
	uint16_t Npixels = (len-4)/16;
	// Another stupid check: len-4 should be divisible by 16 (4bytes/float, 4 heightmap rows)
	if (Npixels*16 != (len-4))
		return;

	hmapdata_["cloud_t"].resize(1);
	memcpy(hmapdata_["cloud_t"].data(), &msg[0], 4);

	Eigen::Map<const Eigen::VectorXf> hmap_data((const float *)&msg[4], Npixels*4);
	hmapdata_["x"] = hmap_data.segment(0,Npixels);
	hmapdata_["z"] = hmap_data.segment(1*Npixels,Npixels);
	hmapdata_["dzdx"] = hmap_data.segment(2*Npixels,Npixels);
	hmapdata_["dzdx_padding"] = hmap_data.segment(3*Npixels,Npixels);

	// std::cout << ctrldata_["z"] << std::endl << std::endl << std::endl;
	accumulateData(rx_accum_, hmapdata_);
}

} // namespace gr

