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
#include "mblink_socket.hpp"
#include <thread>
#include <functional>
#include <mavlink/common/mavlink.h>
#include <Eigen/Core>

namespace gr {

class AppInterface {
public:
#pragma pack(push,1)
	struct MessageFromPlannerToAppHeader_t {
		uint8_t type; //1
		uint8_t res_mm; //2
		int16_t xstart_cm; //4
		uint16_t num_points; //6
		uint8_t mb_rate, cam_rate; //8
	}; // must go in 64 bits = 8 bytes
#pragma pack(pop)
	typedef std::function<void(uint8_t, uint32_t)> SetModeFunc_t;

	~AppInterface() {
		if (rx_thread_.joinable()) {
			rx_thread_.join();
		}
	}

	// Return number of bytes sent
	int send(float res, float xstart, const Eigen::VectorXf &zs, float mb_rate, float cam_rate);

	void rxstart(const SetModeFunc_t &f_set_mode) {
		rx_thread_ = std::thread(&AppInterface::rxThreadTask, this, f_set_mode);
	}
	void rxstop();

private:
	constexpr static int DEBUG_FLOAT_ARRAY_ID_FROM_PLANNER_TO_APP = 1;
	std::string app_IP_ = "127.0.0.1"; // populated after receiving something
	// std::string app_IP_ = "172.20.10.3"; // populated after receiving something
	std::thread rx_thread_;
	SOCKET tx_socket_ = 0, rx_socket_ = 0;
	struct sockaddr_in tx_addr_;
	const char *CLOSE_STR = "close"; // send this secret message

	void rxThreadTask(const SetModeFunc_t &f_set_mode);

	// From app's perspective
	const uint16_t APP_TX_PORT = 14999;
	const uint16_t APP_RX_PORT = 15000;

	// mavlink vars
	mavlink_message_t rx_msg_, tx_msg_;
	mavlink_status_t rx_status_;
	mavlink_set_mode_t set_mode_;
	uint8_t rx_chan_ = 0;
	uint8_t system_id_ = 255, component_id_ = 0;
	int sendMessage(uint16_t msg_len);
};

} // namespace gr
